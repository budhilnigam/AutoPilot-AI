"""
FastAPI Application for AutoPilot AI

Provides REST API and WebSocket endpoints for frontend integration.
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import secrets
import boto3
import json
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from botocore.exceptions import BotoCoreError, ClientError

from config import config
from models.core_models import MetricData, MetricType
from models.agent_protocol import AgentType, Severity
from api.routes import AutoPilotAPI
from services.knowledge_base_factory import KnowledgeBaseInterface
from services.github_service import GitHubService
from services.scheduler import SchedulerService
from services.ops_state_store import OpsStateStore
from services.auth_store import AuthStore

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
scheduler_service = None
ops_state_store = None
auth_store = None


@app.on_event("startup")
async def startup_event():
    """Initialize AutoPilot API on startup"""
    global autopilot_api, github_service, scheduler_service, ops_state_store, auth_store
    logger.info("Starting AutoPilot AI API...")
    autopilot_api = AutoPilotAPI()
    github_service = GitHubService()
    auth_store = AuthStore()

    ops_state_store = OpsStateStore()

    def handle_job_complete(run_result: Dict[str, Any]):
        ops_state_store.append_job_run(run_result)
        job_id = run_result.get('job_id')
        if job_id is not None:
            ops_state_store.update_job(
                int(job_id),
                {
                    'last_run_at': run_result.get('completed_at'),
                    'last_status': run_result.get('status'),
                }
            )

    scheduler_service = SchedulerService(api=autopilot_api, on_job_complete=handle_job_complete)
    scheduler_service.start()
    scheduler_service.sync_autonomous_jobs(ops_state_store.get_jobs())

    logger.info("AutoPilot AI API started successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop background scheduler on shutdown."""
    if scheduler_service:
        scheduler_service.stop()


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
    aws_context: Optional[Dict[str, Any]] = None


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


class ActionCreateRequest(BaseModel):
    title: str


class ActionUpdateRequest(BaseModel):
    status: str


class JobCreateRequest(BaseModel):
    name: str
    prompt: str
    schedule: str


class JobUpdateRequest(BaseModel):
    name: Optional[str] = None
    prompt: Optional[str] = None
    schedule: Optional[str] = None
    enabled: Optional[bool] = None


class RegisterRequest(BaseModel):
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class AwsConnectRequest(BaseModel):
    access_key_id: str
    secret_access_key: str
    session_token: Optional[str] = None
    region: str = 'us-east-1'


class GithubOauthStartRequest(BaseModel):
    repo_owner: Optional[str] = None
    repo_name: Optional[str] = None


class GithubRepoBindingRequest(BaseModel):
    repo_owner: Optional[str] = None
    repo_name: Optional[str] = None


def _extract_bearer_token(authorization: Optional[str]) -> str:
    if not authorization or not authorization.lower().startswith('bearer '):
        raise HTTPException(status_code=401, detail='Missing Bearer token')
    return authorization.split(' ', 1)[1].strip()


def _get_current_user(authorization: Optional[str] = Header(None)) -> Dict[str, Any]:
    token = _extract_bearer_token(authorization)
    user = auth_store.get_user_by_session(token)
    if not user:
        raise HTTPException(status_code=401, detail='Session expired or invalid')
    return user


def _require_aws_connection(user: Dict[str, Any]) -> Dict[str, Any]:
    aws_conn = auth_store.get_aws_connection(user['id'])
    if not aws_conn:
        raise HTTPException(
            status_code=403,
            detail='AWS account connection is mandatory. Connect AWS credentials in Account Settings to continue.',
        )
    return aws_conn


def _require_github_connection(user: Dict[str, Any]) -> Dict[str, Any]:
    github_conn = auth_store.get_github_connection(user['id'])
    if not github_conn:
        raise HTTPException(
            status_code=403,
            detail='GitHub is not connected for this user. Connect GitHub to use CI/CD and repository features.',
        )
    return github_conn


def _oauth_popup_response(status: str, message: str) -> HTMLResponse:
    safe_message = (message or '').replace('"', "'")
    html = f"""
    <!doctype html>
    <html>
      <head><meta charset='utf-8'><title>GitHub OAuth</title></head>
      <body>
        <script>
          (function() {{
            var payload = {{ type: 'github-oauth', status: '{status}', message: '{safe_message}' }};
            if (window.opener && !window.opener.closed) {{
              window.opener.postMessage(payload, '*');
            }}
            window.close();
          }})();
        </script>
        <p>{safe_message}</p>
      </body>
    </html>
    """
    return HTMLResponse(content=html)


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
        aws_context=None,
    )


def _build_aws_runtime_context(aws_connection: Dict[str, Any]) -> Dict[str, Any]:
    """
    Collect minimal live AWS account context for every chat request.

    This enforces account-grounded answers instead of generic LLM replies.
    """
    session = boto3.session.Session(
        aws_access_key_id=aws_connection['access_key_id'],
        aws_secret_access_key=aws_connection['secret_access_key'],
        aws_session_token=aws_connection.get('session_token') or None,
        region_name=aws_connection.get('region') or config.AWS_REGION,
    )

    sts_client = session.client('sts')
    identity = sts_client.get_caller_identity()
    cloudwatch_client = session.client('cloudwatch')
    ce_client = session.client('ce', region_name='us-east-1')

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
            'region': aws_connection.get('region') or config.AWS_REGION,
            'monthly_unblended_cost_usd': total_cost_usd,
            'discovered_ec2_cpu_metrics': discovered_metrics,
            'sampled_at': now.isoformat(),
        }
    }


# ========== Auth and Account Connections ==========

@app.post('/api/auth/register')
async def register(payload: RegisterRequest):
    email = (payload.email or '').strip().lower()
    password = payload.password or ''

    if '@' not in email:
        raise HTTPException(status_code=400, detail='A valid email is required')
    if len(password) < 8:
        raise HTTPException(status_code=400, detail='Password must be at least 8 characters')

    try:
        user = auth_store.create_user(email=email, password=password)
    except Exception as exc:
        if 'UNIQUE constraint failed' in str(exc):
            raise HTTPException(status_code=409, detail='User already exists')
        raise HTTPException(status_code=500, detail='Failed to create user')

    token = auth_store.create_session(user_id=user['id'])
    return {
        'status': 'success',
        'token': token,
        'user': {
            'id': user['id'],
            'email': user['email'],
        },
        'required_connections': {
            'aws': 'mandatory',
            'github': 'optional',
        },
    }


@app.post('/api/auth/login')
async def login(payload: LoginRequest):
    user = auth_store.authenticate(payload.email, payload.password)
    if not user:
        raise HTTPException(status_code=401, detail='Invalid email or password')

    token = auth_store.create_session(user_id=user['id'])
    return {
        'status': 'success',
        'token': token,
        'user': user,
    }


@app.post('/api/auth/logout')
async def logout(authorization: Optional[str] = Header(None)):
    token = _extract_bearer_token(authorization)
    auth_store.revoke_session(token)
    return {'status': 'success'}


@app.get('/api/auth/me')
async def me(user: Dict[str, Any] = Depends(_get_current_user)):
    return {
        'status': 'success',
        'user': user,
        'connections': auth_store.get_connection_status(user['id']),
    }


@app.post('/api/auth/connect/aws')
async def connect_aws(payload: AwsConnectRequest, user: Dict[str, Any] = Depends(_get_current_user)):
    access_key_id = (payload.access_key_id or '').strip()
    secret_access_key = (payload.secret_access_key or '').strip()
    region = (payload.region or config.AWS_REGION).strip()

    if not access_key_id or not secret_access_key:
        raise HTTPException(status_code=400, detail='AWS access key id and secret access key are required')

    session = boto3.session.Session(
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        aws_session_token=payload.session_token,
        region_name=region,
    )

    permissions = {
        'sts:GetCallerIdentity': False,
        'cloudwatch:ListMetrics': False,
        'ce:GetCostAndUsage': False,
    }

    identity = None
    try:
        identity = session.client('sts').get_caller_identity()
        permissions['sts:GetCallerIdentity'] = True
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f'AWS validation failed: {exc}')

    try:
        session.client('cloudwatch').list_metrics(Namespace='AWS/EC2', MetricName='CPUUtilization')
        permissions['cloudwatch:ListMetrics'] = True
    except Exception:
        logger.warning('AWS credential lacks cloudwatch:ListMetrics')

    try:
        now = datetime.utcnow()
        session.client('ce', region_name='us-east-1').get_cost_and_usage(
            TimePeriod={'Start': now.replace(day=1).date().strftime('%Y-%m-%d'), 'End': now.date().strftime('%Y-%m-%d')},
            Granularity='MONTHLY',
            Metrics=['UnblendedCost'],
        )
        permissions['ce:GetCostAndUsage'] = True
    except Exception:
        logger.warning('AWS credential lacks ce:GetCostAndUsage')

    auth_store.upsert_aws_connection(
        user_id=user['id'],
        account_id=identity.get('Account', ''),
        arn=identity.get('Arn', ''),
        region=region,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=payload.session_token,
        validated_permissions=json.dumps(permissions),
    )

    return {
        'status': 'success',
        'message': 'AWS account connected',
        'account_id': identity.get('Account'),
        'arn': identity.get('Arn'),
        'region': region,
        'required_permissions': list(permissions.keys()),
        'granted_permissions': [name for name, granted in permissions.items() if granted],
        'missing_permissions': [name for name, granted in permissions.items() if not granted],
    }


@app.delete('/api/auth/connect/aws')
async def disconnect_aws(user: Dict[str, Any] = Depends(_get_current_user)):
    auth_store.delete_aws_connection(user['id'])
    return {'status': 'success', 'message': 'AWS account disconnected'}


@app.post('/api/auth/connect/github/oauth/start')
async def start_github_oauth(payload: GithubOauthStartRequest, user: Dict[str, Any] = Depends(_get_current_user)):
    if not config.GITHUB_OAUTH_CLIENT_ID or not config.GITHUB_OAUTH_CLIENT_SECRET:
        raise HTTPException(status_code=500, detail='GitHub OAuth is not configured on server')

    redirect_uri = config.GITHUB_OAUTH_REDIRECT_URI or 'http://localhost:8000/api/auth/connect/github/oauth/callback'

    state = secrets.token_urlsafe(24)
    repo_owner = (payload.repo_owner or '').strip() or None
    repo_name = (payload.repo_name or '').strip() or None
    auth_store.create_github_oauth_state(
        user_id=user['id'],
        state=state,
        repo_owner=repo_owner,
        repo_name=repo_name,
    )

    required_scopes = ['repo', 'workflow', 'read:user']
    auth_params = {
        'client_id': config.GITHUB_OAUTH_CLIENT_ID,
        'redirect_uri': redirect_uri,
        'scope': ' '.join(required_scopes),
        'state': state,
        'allow_signup': 'false',
    }
    auth_url = requests.Request('GET', 'https://github.com/login/oauth/authorize', params=auth_params).prepare().url

    return {
        'status': 'success',
        'auth_url': auth_url,
        'state': state,
        'required_scopes': required_scopes,
        'minimal_scopes_policy': [
            'repo: repository metadata and commits/workflows access',
            'workflow: read CI workflow runs and health',
            'read:user: identify the connected account',
        ],
    }


@app.get('/api/auth/connect/github/oauth/callback')
async def github_oauth_callback(code: Optional[str] = None, state: Optional[str] = None, error: Optional[str] = None):
    if error:
        return _oauth_popup_response('error', f'GitHub authorization failed: {error}')
    if not code or not state:
        return _oauth_popup_response('error', 'GitHub authorization failed: missing code or state')

    state_row = auth_store.consume_github_oauth_state(state)
    if not state_row:
        return _oauth_popup_response('error', 'GitHub authorization failed: state is invalid or expired')

    redirect_uri = config.GITHUB_OAUTH_REDIRECT_URI or 'http://localhost:8000/api/auth/connect/github/oauth/callback'
    token_res = requests.post(
        'https://github.com/login/oauth/access_token',
        headers={
            'Accept': 'application/json',
            'User-Agent': 'autopilot-ai',
        },
        data={
            'client_id': config.GITHUB_OAUTH_CLIENT_ID,
            'client_secret': config.GITHUB_OAUTH_CLIENT_SECRET,
            'code': code,
            'redirect_uri': redirect_uri,
            'state': state,
        },
        timeout=20,
    )

    if token_res.status_code >= 400:
        return _oauth_popup_response('error', 'GitHub token exchange failed')

    token_payload = token_res.json()
    access_token = token_payload.get('access_token')
    if not access_token:
        return _oauth_popup_response('error', 'GitHub token exchange failed: access token missing')

    gh_headers = {
        'Authorization': f'Bearer {access_token}',
        'Accept': 'application/vnd.github+json',
        'User-Agent': 'autopilot-ai',
    }
    user_res = requests.get('https://api.github.com/user', headers=gh_headers, timeout=20)
    if user_res.status_code >= 400:
        return _oauth_popup_response('error', 'GitHub user validation failed')

    gh_user = user_res.json()
    granted_raw = user_res.headers.get('X-OAuth-Scopes') or token_payload.get('scope', '')
    granted = {item.strip() for item in str(granted_raw).split(',') if item.strip()}

    auth_store.upsert_github_connection(
        user_id=state_row['user_id'],
        username=gh_user.get('login', ''),
        token=access_token,
        scopes=','.join(sorted(granted)),
        repo_owner=state_row.get('repo_owner'),
        repo_name=state_row.get('repo_name'),
    )

    return _oauth_popup_response('success', 'GitHub account connected successfully')


@app.post('/api/auth/connect/github/repository')
async def bind_github_repository(payload: GithubRepoBindingRequest, user: Dict[str, Any] = Depends(_get_current_user)):
    _require_github_connection(user)
    auth_store.update_github_repository(
        user_id=user['id'],
        repo_owner=(payload.repo_owner or '').strip() or None,
        repo_name=(payload.repo_name or '').strip() or None,
    )
    return {'status': 'success', 'message': 'GitHub repository preference updated'}


@app.delete('/api/auth/connect/github')
async def disconnect_github(user: Dict[str, Any] = Depends(_get_current_user)):
    auth_store.delete_github_connection(user['id'])
    return {'status': 'success', 'message': 'GitHub account disconnected'}


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
async def chat(message: ChatMessage, user: Dict[str, Any] = Depends(_get_current_user)):
    """
    Main chat endpoint for SRE AI Copilot.
    
    Processes user queries through the multi-agent system.
    """
    try:
        start_time = datetime.utcnow()

        aws_connection = _require_aws_connection(user)

        try:
            aws_context = _build_aws_runtime_context(aws_connection)
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
                            'cost_impact': insight.get('cost_impact'),
                            'evidence': insight.get('evidence') or insight.get('details') or {},
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
                            'cost_impact': getattr(insight, 'cost_impact', None),
                            'evidence': getattr(insight, 'evidence', {}),
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
            execution_time_ms=execution_time,
            aws_context=aws_context.get('aws_account'),
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

def _user_github_service(user: Dict[str, Any]) -> tuple[GitHubService, Optional[str], Optional[str]]:
    github_conn = _require_github_connection(user)
    service = GitHubService(token=github_conn['token'])
    return service, github_conn.get('repo_owner'), github_conn.get('repo_name')

@app.get("/api/github/repo/info")
async def get_github_repo_info(user: Dict[str, Any] = Depends(_get_current_user)):
    """
    Get repository information.
    Requires GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME in configuration.
    """
    try:
        service, owner, repo = _user_github_service(user)
        repo_info = service.get_repository_info(owner=owner, repo=repo)
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
async def get_recent_builds(limit: int = 10, user: Dict[str, Any] = Depends(_get_current_user)):
    """
    Get recent workflow runs.
    """
    try:
        service, owner, repo = _user_github_service(user)
        builds = service.get_recent_builds(limit=min(limit, 50), owner=owner, repo=repo)
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
async def get_failed_builds(limit: int = 5, user: Dict[str, Any] = Depends(_get_current_user)):
    """
    Get recent failed workflow runs.
    """
    try:
        service, owner, repo = _user_github_service(user)
        failed_builds = service.get_failed_builds(limit=min(limit, 20), owner=owner, repo=repo)
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
async def get_build_trends(days: int = 7, user: Dict[str, Any] = Depends(_get_current_user)):
    """
    Get build trend analysis.
    """
    try:
        service, owner, repo = _user_github_service(user)
        trends = service.get_build_trends(days=days, owner=owner, repo=repo)
        return trends
    except Exception as e:
        logger.error(f"Build trends error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/builds/health")
async def get_build_health(user: Dict[str, Any] = Depends(_get_current_user)):
    """
    Get overall build health summary.
    """
    try:
        service, owner, repo = _user_github_service(user)
        health = service.get_build_health_summary(owner=owner, repo=repo)
        return health
    except Exception as e:
        logger.error(f"Build health error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/github/commits")
async def get_commits(branch: str = "main", limit: int = 20, user: Dict[str, Any] = Depends(_get_current_user)):
    """
    Get commit history for a branch.
    """
    try:
        service, owner, repo = _user_github_service(user)
        commits = service.get_commit_history(branch=branch, limit=min(limit, 50), owner=owner, repo=repo)
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
async def get_workflow_runs(limit: int = 10, user: Dict[str, Any] = Depends(_get_current_user)):
    """
    Get recent workflow runs.
    """
    try:
        service, owner, repo = _user_github_service(user)
        runs = service.get_workflow_runs(limit=min(limit, 50), owner=owner, repo=repo)
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


# ========== Ops Center Endpoints ==========

@app.get("/api/ops/actions")
async def get_ops_actions(user: Dict[str, Any] = Depends(_get_current_user)):
    """Get persisted Action Center items."""
    _require_aws_connection(user)
    return {
        'status': 'success',
        'actions': ops_state_store.get_actions(),
    }


@app.post("/api/ops/actions")
async def create_ops_action(payload: ActionCreateRequest, user: Dict[str, Any] = Depends(_get_current_user)):
    """Create a new Action Center item."""
    _require_aws_connection(user)
    title = (payload.title or '').strip()
    if not title:
        raise HTTPException(status_code=400, detail='Action title cannot be empty')

    action = ops_state_store.create_action(title)
    return {
        'status': 'success',
        'action': action,
    }


@app.patch("/api/ops/actions/{action_id}")
async def update_ops_action(action_id: int, payload: ActionUpdateRequest, user: Dict[str, Any] = Depends(_get_current_user)):
    """Update action status."""
    _require_aws_connection(user)
    allowed = {'Suggested', 'Approved', 'Executed', 'Verified'}
    if payload.status not in allowed:
        raise HTTPException(status_code=400, detail=f'Invalid status. Allowed: {sorted(allowed)}')

    updated = ops_state_store.update_action(action_id, payload.status)
    if not updated:
        raise HTTPException(status_code=404, detail='Action not found')

    return {
        'status': 'success',
        'action': updated,
    }


@app.get("/api/ops/jobs")
async def get_ops_jobs(user: Dict[str, Any] = Depends(_get_current_user)):
    """Get persisted autonomous jobs."""
    _require_aws_connection(user)
    return {
        'status': 'success',
        'jobs': ops_state_store.get_jobs(),
    }


@app.post("/api/ops/jobs")
async def create_ops_job(payload: JobCreateRequest, user: Dict[str, Any] = Depends(_get_current_user)):
    """Create and schedule a new autonomous job."""
    _require_aws_connection(user)
    name = (payload.name or '').strip()
    prompt = (payload.prompt or '').strip()
    cadence = (payload.schedule or '').strip().lower()
    if not name or not prompt:
        raise HTTPException(status_code=400, detail='Job name and prompt are required')
    if cadence not in {'hourly', 'daily', 'weekly'}:
        raise HTTPException(status_code=400, detail='Job schedule must be hourly, daily, or weekly')

    job = ops_state_store.create_job(name=name, prompt=prompt, schedule=cadence)
    scheduler_service.register_autonomous_job(job)

    return {
        'status': 'success',
        'job': job,
    }


@app.patch("/api/ops/jobs/{job_id}")
async def update_ops_job(job_id: int, payload: JobUpdateRequest, user: Dict[str, Any] = Depends(_get_current_user)):
    """Update and re-sync autonomous job."""
    _require_aws_connection(user)
    fields: Dict[str, Any] = {}
    if payload.name is not None:
        fields['name'] = payload.name.strip()
    if payload.prompt is not None:
        fields['prompt'] = payload.prompt.strip()
    if payload.schedule is not None:
        cadence = payload.schedule.strip().lower()
        if cadence not in {'hourly', 'daily', 'weekly'}:
            raise HTTPException(status_code=400, detail='Job schedule must be hourly, daily, or weekly')
        fields['schedule'] = cadence
    if payload.enabled is not None:
        fields['enabled'] = payload.enabled

    updated = ops_state_store.update_job(job_id, fields)
    if not updated:
        raise HTTPException(status_code=404, detail='Job not found')

    scheduler_service.register_autonomous_job(updated)

    return {
        'status': 'success',
        'job': updated,
    }


@app.post("/api/ops/jobs/{job_id}/run")
async def run_ops_job(job_id: int, user: Dict[str, Any] = Depends(_get_current_user)):
    """Run an autonomous job immediately."""
    _require_aws_connection(user)
    jobs = ops_state_store.get_jobs()
    job = next((item for item in jobs if item.get('id') == job_id), None)
    if not job:
        raise HTTPException(status_code=404, detail='Job not found')

    run_result = scheduler_service.trigger_job_now(job)

    # Keep a fresh copy of last run metadata in job state.
    ops_state_store.update_job(
        job_id,
        {
            'last_run_at': run_result.get('completed_at'),
            'last_status': run_result.get('status'),
        }
    )

    return {
        'status': 'success',
        'run': run_result,
    }


@app.get("/api/ops/jobs/runs")
async def list_ops_job_runs(limit: int = 20, user: Dict[str, Any] = Depends(_get_current_user)):
    """Return recent autonomous job executions."""
    _require_aws_connection(user)
    return {
        'status': 'success',
        'runs': ops_state_store.get_job_runs(limit=limit),
    }
