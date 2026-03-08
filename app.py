"""
FastAPI Application for AutoPilot AI

Provides REST API and WebSocket endpoints for frontend integration.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import boto3
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from botocore.exceptions import BotoCoreError, ClientError

from config import config
from models.core_models import MetricData, MetricType
from models.agent_protocol import AgentType, Severity
from api.routes import AutoPilotAPI
from services.knowledge_base_factory import KnowledgeBaseInterface
from services.github_service import GitHubService

logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="AutoPilot AI",
    description="Multi-Agent AI SRE System",
    version="1.0.0"
)

# CORS middleware
if config.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_origin_regex=config.CORS_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Initialize AutoPilot API
autopilot_api = None
github_service = None


@app.on_event("startup")
async def startup_event():
    """Initialize AutoPilot API on startup"""
    global autopilot_api, github_service
    logger.info("Starting AutoPilot AI API...")
    autopilot_api = AutoPilotAPI()
    github_service = GitHubService()
    logger.info("AutoPilot AI API started successfully")


# ========== Pydantic Models ==========

class ChatMessage(BaseModel):
    """Chat message from user"""
    message: str
    context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    """Chat response to user"""
    response: str
    thinking: Optional[str] = None
    insights: List[Dict[str, Any]] = []
    recommendations: List[str] = []
    agent_type: Optional[str] = None
    execution_time_ms: float = 0


class HealthCheckResponse(BaseModel):
    """Health check response"""
    service: str
    status: str  # "healthy", "degraded", "unhealthy"
    message: str
    timestamp: str
    response_time_ms: Optional[float] = None


class DefaultPrompt(BaseModel):
    """Default prompt for quick access"""
    id: str
    title: str
    prompt: str
    category: str
    icon: str


def _aws_unavailable_chat_response(start_time: datetime, error_message: str) -> ChatResponse:
    """Return a strict, concise chat response when AWS account context cannot be fetched."""
    execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
    return ChatResponse(
        response=(
            "Unable to answer from AWS account data right now because AWS connectivity/authentication failed: "
            f"{error_message}"
        ),
        thinking=None,
        insights=[],
        recommendations=[],
        agent_type="system",
        execution_time_ms=execution_time,
    )


def _build_aws_runtime_context() -> Dict[str, Any]:
    """
    Collect minimal live AWS account context for every chat request.

    This enforces account-grounded answers instead of generic LLM replies.
    """
    # Validate current AWS identity first.
    sts_client = boto3.client('sts', region_name=config.AWS_REGION)
    identity = sts_client.get_caller_identity()

    cloudwatch_client = autopilot_api.cloudwatch_client.cloudwatch
    ce_client = autopilot_api.billing_client.ce_client

    now = datetime.utcnow()
    month_start = now.replace(day=1).date().strftime('%Y-%m-%d')
    today = now.date().strftime('%Y-%m-%d')

    ce_response = ce_client.get_cost_and_usage(
        TimePeriod={
            'Start': month_start,
            'End': today,
        },
        Granularity='MONTHLY',
        Metrics=['UnblendedCost'],
    )

    metrics_response = cloudwatch_client.list_metrics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
    )

    total_cost_usd = 0.0
    results_by_time = ce_response.get('ResultsByTime', [])
    if results_by_time:
        total_cost_usd = float(
            results_by_time[0]
            .get('Total', {})
            .get('UnblendedCost', {})
            .get('Amount', '0')
        )

    discovered_metrics = len(metrics_response.get('Metrics', []))

    return {
        'aws_account': {
            'account_id': identity.get('Account'),
            'arn': identity.get('Arn'),
            'region': config.AWS_REGION,
            'monthly_unblended_cost_usd': total_cost_usd,
            'discovered_ec2_cpu_metrics': discovered_metrics,
            'sampled_at': now.isoformat(),
        }
    }


# ========== Health Check Endpoints ==========

@app.get("/api/health")
async def health_check():
    """Overall system health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }


@app.get("/api/health/agents", response_model=List[HealthCheckResponse])
async def health_check_agents():
    """Check health of active agents"""
    
    agents_to_check = [
        ("Planner Agent", autopilot_api.planner_agent),
        ("Unified Agent", autopilot_api.unified_agent),
    ]
    
    results = []
    for name, agent in agents_to_check:
        start_time = datetime.utcnow()
        try:
            # Simple check - agent should be initialized
            if agent is None:
                results.append(HealthCheckResponse(
                    service=name,
                    status="unhealthy",
                    message="Agent not initialized",
                    timestamp=datetime.utcnow().isoformat()
                ))
            else:
                response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
                results.append(HealthCheckResponse(
                    service=name,
                    status="healthy",
                    message="Agent operational",
                    timestamp=datetime.utcnow().isoformat(),
                    response_time_ms=response_time
                ))
        except Exception as e:
            results.append(HealthCheckResponse(
                service=name,
                status="unhealthy",
                message=str(e),
                timestamp=datetime.utcnow().isoformat()
            ))
    
    return results


@app.get("/api/health/services", response_model=List[HealthCheckResponse])
async def health_check_services():
    """Check health of all services"""
    
    results = []
    
    # Check Bedrock
    start_time = datetime.utcnow()
    try:
        # Simple check
        if autopilot_api.bedrock_client:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            results.append(HealthCheckResponse(
                service="Bedrock Client",
                status="healthy",
                message="Bedrock client initialized",
                timestamp=datetime.utcnow().isoformat(),
                response_time_ms=response_time
            ))
        else:
            results.append(HealthCheckResponse(
                service="Bedrock Client",
                status="unhealthy",
                message="Bedrock client not initialized",
                timestamp=datetime.utcnow().isoformat()
            ))
    except Exception as e:
        results.append(HealthCheckResponse(
            service="Bedrock Client",
            status="unhealthy",
            message=str(e),
            timestamp=datetime.utcnow().isoformat()
        ))
    
    # Check Knowledge Base
    start_time = datetime.utcnow()
    try:
        kb = KnowledgeBaseInterface()
        is_healthy = kb.health_check()
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        results.append(HealthCheckResponse(
            service="Knowledge Base",
            status="healthy" if is_healthy else "degraded",
            message="Local KB" if kb.is_local else "AWS KB",
            timestamp=datetime.utcnow().isoformat(),
            response_time_ms=response_time
        ))
    except Exception as e:
        results.append(HealthCheckResponse(
            service="Knowledge Base",
            status="unhealthy",
            message=str(e),
            timestamp=datetime.utcnow().isoformat()
        ))
    
    # Check CloudWatch
    start_time = datetime.utcnow()
    try:
        if autopilot_api.cloudwatch_client:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            results.append(HealthCheckResponse(
                service="CloudWatch",
                status="healthy",
                message="CloudWatch client initialized",
                timestamp=datetime.utcnow().isoformat(),
                response_time_ms=response_time
            ))
        else:
            results.append(HealthCheckResponse(
                service="CloudWatch",
                status="degraded",
                message="CloudWatch client not initialized",
                timestamp=datetime.utcnow().isoformat()
            ))
    except Exception as e:
        results.append(HealthCheckResponse(
            service="CloudWatch",
            status="unhealthy",
            message=str(e),
            timestamp=datetime.utcnow().isoformat()
        ))
    
    # Check GitHub
    start_time = datetime.utcnow()
    try:
        if autopilot_api.github_client and autopilot_api.github_client.token:
            response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            results.append(HealthCheckResponse(
                service="GitHub",
                status="healthy",
                message="GitHub client connected",
                timestamp=datetime.utcnow().isoformat(),
                response_time_ms=response_time
            ))
        else:
            results.append(HealthCheckResponse(
                service="GitHub",
                status="degraded",
                message="GitHub client not fully configured",
                timestamp=datetime.utcnow().isoformat()
            ))
    except Exception as e:
        results.append(HealthCheckResponse(
            service="GitHub",
            status="unhealthy",
            message=str(e),
            timestamp=datetime.utcnow().isoformat()
        ))
    
    return results


# ========== Chat Endpoint ==========

@app.post("/api/chat", response_model=ChatResponse)
async def chat(message: ChatMessage):
    """
    Main chat endpoint for SRE AI Copilot.
    
    Processes user queries through the multi-agent system.
    """
    try:
        start_time = datetime.utcnow()

        try:
            aws_context = _build_aws_runtime_context()
        except (ClientError, BotoCoreError, Exception) as aws_error:
            logger.warning(f"AWS context fetch failed for chat: {aws_error}")
            return _aws_unavailable_chat_response(start_time, str(aws_error))

        merged_context = dict(message.context or {})
        merged_context.update(aws_context)
        
        # Process query through Planner Agent
        result = autopilot_api.query(message.message, merged_context)
        
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Extract insights and recommendations
        insights = []
        recommendations = []
        
        if result.get('insights'):
            for insight in result.get('insights', []):
                if isinstance(insight, dict):
                    severity_value = insight.get('severity', 'medium')
                    insights.append(
                        {
                            'summary': insight.get('summary', 'Insight generated'),
                            'severity': str(severity_value).lower(),
                            'business_impact': insight.get('business_impact', ''),
                            'confidence_score': insight.get('confidence') or insight.get('confidence_score', 0.0),
                        }
                    )
                    recommendations.extend(insight.get('recommendations', []))
                else:
                    severity_attr = getattr(insight, 'severity', 'medium')
                    severity_value = severity_attr.value if hasattr(severity_attr, 'value') else str(severity_attr)
                    insights.append(
                        {
                            'summary': getattr(insight, 'summary', 'Insight generated'),
                            'severity': str(severity_value).lower(),
                            'business_impact': getattr(insight, 'business_impact', ''),
                            'confidence_score': getattr(insight, 'confidence_score', 0.0),
                        }
                    )
                    recommendations.extend(getattr(insight, 'recommendations', []))
        
        aws_summary_hint = (
            f"AWS Account {aws_context['aws_account']['account_id']} "
            f"({aws_context['aws_account']['region']}) | "
            f"Monthly cost (USD): {aws_context['aws_account']['monthly_unblended_cost_usd']:.2f} | "
            f"Discovered EC2 CPU metrics: {aws_context['aws_account']['discovered_ec2_cpu_metrics']}"
        )

        response_text = (
            result.get('summary')
            or result.get('response')
            or result.get('content')
        )

        thinking_parts: List[str] = []
        result_data = result.get('data', {})
        if isinstance(result_data, dict):
            for payload in result_data.values():
                if isinstance(payload, dict):
                    thought = (payload.get('thinking') or '').strip()
                    if thought:
                        thinking_parts.append(thought)

        deduped_thinking = []
        seen = set()
        for thought in thinking_parts:
            if thought not in seen:
                deduped_thinking.append(thought)
                seen.add(thought)

        thinking_text = "\n\n".join(deduped_thinking) if deduped_thinking else None

        # Removed fallback that duplicated insights into response_text
        # The planner now properly synthesizes response from agent data

        if not response_text and result.get('status') == 'FAILED':
            response_text = result.get('error') or 'Request processing failed in the planner pipeline.'

        if not response_text:
            response_text = f"AWS-grounded analysis completed. {aws_summary_hint}"

        return ChatResponse(
            response=response_text,
            thinking=thinking_text,
            insights=insights,
            recommendations=list(set(recommendations)),
            agent_type=result.get('agent_type'),
            execution_time_ms=execution_time
        )
        
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Default Prompts ==========

@app.get("/api/prompts/default", response_model=List[DefaultPrompt])
async def get_default_prompts():
    """Get default prompts for quick access"""
    
    prompts = [
        DefaultPrompt(
            id="analyze-metrics",
            title="Analyze Recent Metrics",
            prompt="Analyze the recent CloudWatch metrics and identify any anomalies or performance issues",
            category="observability",
            icon="📊"
        ),
        DefaultPrompt(
            id="cost-optimization",
            title="Cost Optimization",
            prompt="Analyze current AWS costs and provide optimization recommendations in INR",
            category="cost",
            icon="💰"
        ),
        DefaultPrompt(
            id="infrastructure-health",
            title="Infrastructure Health Check",
            prompt="Check the health of our ECS services and container configurations",
            category="infrastructure",
            icon="🏗️"
        ),
        DefaultPrompt(
            id="database-performance",
            title="Database Performance",
            prompt="Analyze database query performance and suggest index optimizations",
            category="database",
            icon="🗄️"
        ),
        DefaultPrompt(
            id="cicd-analysis",
            title="CI/CD Build Analysis",
            prompt="Analyze recent CI/CD builds and identify any regressions or failures",
            category="cicd",
            icon="🚀"
        ),
        DefaultPrompt(
            id="predict-saturation",
            title="Predict Resource Saturation",
            prompt="Predict when our resources might reach saturation based on current trends",
            category="predictive",
            icon="🔮"
        ),
        DefaultPrompt(
            id="redis-analysis",
            title="Redis Performance",
            prompt="Analyze Redis performance metrics and memory usage patterns",
            category="database",
            icon="⚡"
        ),
        DefaultPrompt(
            id="dockerfile-review",
            title="Dockerfile Review",
            prompt="Review our Dockerfile for optimization opportunities and best practices",
            category="infrastructure",
            icon="🐳"
        ),
    ]
    
    return prompts


# ========== WebSocket for Live Alerts ==========

class ConnectionManager:
    """Manage WebSocket connections for live alerts"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def broadcast(self, message: dict):
        """Broadcast message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error sending to websocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


manager = ConnectionManager()


@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """
    WebSocket endpoint for real-time alerts.
    
    Sends live alerts from CloudWatch, cost anomalies, etc.
    """
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive and listen for client messages
            data = await websocket.receive_text()
            
            # Could handle client commands here
            if data == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def send_alert(alert: dict):
    """
    Send an alert to all connected WebSocket clients.
    
    Called by the scheduler or alert service.
    """
    await manager.broadcast({
        "type": "alert",
        "data": alert,
        "timestamp": datetime.utcnow().isoformat()
    })


# ========== GitHub Integration Endpoints ==========

@app.get("/api/github/repo/info")
async def get_github_repo_info():
    """
    Get repository information.
    Requires GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in configuration.
    """
    if not github_service or not github_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in .env"
        )
    
    try:
        repo_info = github_service.get_repository_info()
        if repo_info:
            return {
                "status": "success",
                "repository": repo_info.get('full_name'),
                "description": repo_info.get('description'),
                "url": repo_info.get('html_url'),
                "stars": repo_info.get('stargazers_count'),
                "language": repo_info.get('language'),
                "last_updated": repo_info.get('updated_at')
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve repository info")
    except Exception as e:
        logger.error(f"GitHub repo info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/builds/recent")
async def get_recent_builds(limit: int = 10):
    """
    Get recent workflow runs.
    """
    if not github_service or not github_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in .env"
        )
    
    try:
        builds = github_service.get_recent_builds(limit=min(limit, 50))
        return {
            "status": "success",
            "total_builds": len(builds),
            "builds": [
                {
                    "build_id": b.build_id,
                    "status": b.status,
                    "branch": b.branch,
                    "commit_sha": b.commit_sha[:7],
                    "build_time_seconds": b.build_time_seconds,
                    "timestamp": b.timestamp,
                    "repository": b.repository
                }
                for b in builds
            ]
        }
    except Exception as e:
        logger.error(f"Recent builds error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/builds/failed")
async def get_failed_builds(limit: int = 5):
    """
    Get recent failed workflow runs.
    """
    if not github_service or not github_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in .env"
        )
    
    try:
        failed_builds = github_service.get_failed_builds(limit=min(limit, 20))
        return {
            "status": "success",
            "total_failed": len(failed_builds),
            "builds": [
                {
                    "build_id": b.build_id,
                    "status": b.status,
                    "branch": b.branch,
                    "commit_sha": b.commit_sha[:7],
                    "build_time_seconds": b.build_time_seconds,
                    "timestamp": b.timestamp,
                    "steps": b.steps
                }
                for b in failed_builds
            ]
        }
    except Exception as e:
        logger.error(f"Failed builds error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/builds/trends")
async def get_build_trends(days: int = 7):
    """
    Get build trend analysis.
    """
    if not github_service or not github_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in .env"
        )
    
    try:
        trends = github_service.get_build_trends(days=days)
        return trends
    except Exception as e:
        logger.error(f"Build trends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/builds/health")
async def get_build_health():
    """
    Get overall build health summary.
    """
    if not github_service or not github_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in .env"
        )
    
    try:
        health = github_service.get_build_health_summary()
        return health
    except Exception as e:
        logger.error(f"Build health error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/commits")
async def get_commits(branch: str = "main", limit: int = 20):
    """
    Get commit history for a branch.
    """
    if not github_service or not github_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in .env"
        )
    
    try:
        commits = github_service.get_commit_history(branch=branch, limit=min(limit, 50))
        return {
            "status": "success",
            "branch": branch,
            "total_commits": len(commits),
            "commits": [
                {
                    "sha": c.get('sha', '')[:7],
                    "message": c.get('commit', {}).get('message', '').split('\n')[0],
                    "author": c.get('commit', {}).get('author', {}).get('name', 'Unknown'),
                    "date": c.get('commit', {}).get('author', {}).get('date', ''),
                    "url": c.get('html_url', '')
                }
                for c in commits
            ]
        }
    except Exception as e:
        logger.error(f"Commits history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/workflows")
async def get_workflow_runs(limit: int = 10):
    """
    Get recent workflow runs.
    """
    if not github_service or not github_service.is_configured():
        raise HTTPException(
            status_code=400,
            detail="GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in .env"
        )
    
    try:
        runs = github_service.get_workflow_runs(limit=min(limit, 50))
        return {
            "status": "success",
            "total_runs": len(runs),
            "runs": [
                {
                    "id": r.get('id'),
                    "name": r.get('name'),
                    "status": r.get('status'),
                    "conclusion": r.get('conclusion'),
                    "event": r.get('event'),
                    "branch": r.get('head_branch'),
                    "created_at": r.get('created_at'),
                    "updated_at": r.get('updated_at'),
                    "run_number": r.get('run_number')
                }
                for r in runs
            ]
        }
    except Exception as e:
        logger.error(f"Workflow runs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ========== Additional Endpoints ==========

@app.get("/api/config")
async def get_config():
    """Get safe configuration summary for frontend"""
    return config.get_summary()


@app.get("/api/agents")
async def list_agents():
    """List all available agents"""
    return [
        {"type": "planner", "name": "Planner Agent", "description": "Orchestrates multi-agent workflows"},
        {"type": "observability", "name": "Observability Agent", "description": "Metric analysis and anomaly detection"},
        {"type": "infrastructure", "name": "Infrastructure Agent", "description": "Infrastructure configuration analysis"},
        {"type": "database", "name": "Database Agent", "description": "Database performance optimization"},
        {"type": "cost", "name": "Cost Agent", "description": "AWS cost analysis and optimization"},
        {"type": "cicd", "name": "CI/CD Agent", "description": "Build analysis and regression detection"},
    ]
