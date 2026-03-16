"""
Observability Agent

Responsible for:
- Metric interpretation with business context
- Anomaly detection using statistical analysis
- Bottleneck attribution
- Semantic insight generation
"""

import logging
import statistics
import time
import json
import re
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from models.agent_protocol import AgentResponse, AgentType, TaskStatus, Severity, Insight
from models.core_models import MetricData, Anomaly, Recommendation
from services.bedrock_client import BedrockClient
from services.knowledge_base import KnowledgeBase
from services.cloudwatch_client import CloudWatchClient
from services.aws_api_executor import get_aws_executor
from services.github_service import GitHubService
from config import config

logger = logging.getLogger(__name__)


class ObservabilityAgent:
    """
    Observability Agent for metric analysis and anomaly detection.
    
    Uses dynamic AWS API execution via tool calling for maximum flexibility.
    """
    
    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        knowledge_base: KnowledgeBase = None,
        cloudwatch_client: CloudWatchClient = None,
        github_service: GitHubService = None,
    ):
        """
        Initialize Observability Agent.
        
        Args:
            bedrock_client: Bedrock client for LLM
            knowledge_base: Knowledge base for RAG
            cloudwatch_client: CloudWatch client for metrics (legacy, kept for compatibility)
            github_service: GitHub service for repository and CI/CD data
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.knowledge_base = knowledge_base
        self.cloudwatch_client = cloudwatch_client or CloudWatchClient()
        self.github_service = github_service or GitHubService()
        self.agent_type = AgentType.OBSERVABILITY
        
        # Get the dynamic AWS executor
        self.aws_executor = get_aws_executor(config.AWS_REGION)
        
        logger.info("Observability Agent initialized with dynamic AWS API executor and GitHub integration")

    def _get_github_tool_definition(self) -> Dict[str, Any]:
        """
        Get the GitHub tool definition for LLM function calling.
        
        Returns:
            Tool definition in Anthropic tool format
        """
        return {
            "name": "github_query",
            "description": (
                "Query GitHub repository data including repository info, builds, commits, and CI/CD metrics. "
                "You can target any repository the token can access (including private repos)."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": (
                            "GitHub operation to perform. Options: "
                            "'get_repo_info' (repository metadata), "
                            "'get_recent_builds' (recent workflow runs), "
                            "'get_failed_builds' (failed workflow runs), "
                            "'get_build_trends' (build success/failure trends), "
                            "'get_recent_commits' (recent commit history), "
                            "'get_workflows' (available GitHub Actions workflows), "
                            "'get_accessible_repos' (list repos accessible by token, including private)"
                        ),
                        "enum": [
                            "get_repo_info",
                            "get_recent_builds", 
                            "get_failed_builds",
                            "get_build_trends",
                            "get_recent_commits",
                            "get_workflows",
                            "get_accessible_repos"
                        ]
                    },
                    "parameters": {
                        "type": "object",
                        "description": (
                            "Optional parameters for the operation. "
                            "Target repository with either {'owner': 'org-or-user', 'repo': 'repo-name'} "
                            "or {'repository': 'owner/repo'}. "
                            "For get_recent_builds/get_failed_builds: {'limit': 10}. "
                            "For get_build_trends: {'days': 7}. "
                            "For get_recent_commits: {'limit': 20, 'branch': 'main'}. "
                            "For get_accessible_repos: {'limit': 20}."
                        )
                    }
                },
                "required": ["operation"]
            }
        }

    def _execute_github_operation(self, operation: str, parameters: Dict[str, Any] = None) -> Any:
        """
        Execute a GitHub operation.
        
        Args:
            operation: GitHub operation name
            parameters: Optional parameters for the operation
            
        Returns:
            Operation result
        """
        if not self.github_service.is_configured():
            return {"error": "GitHub token is not configured. Set GITHUB_TOKEN in .env"}
        
        parameters = parameters or {}

        repo_ref = parameters.get('repository')
        owner = parameters.get('owner')
        repo = parameters.get('repo')
        if repo_ref and isinstance(repo_ref, str) and '/' in repo_ref:
            owner, repo = repo_ref.split('/', 1)
        
        try:
            if operation == "get_accessible_repos":
                limit = parameters.get('limit', 20)
                return self.github_service.list_accessible_repositories(limit=limit)

            if operation == "get_repo_info":
                return self.github_service.get_repository_info(owner=owner, repo=repo)
            
            elif operation == "get_recent_builds":
                limit = parameters.get('limit', 10)
                builds = self.github_service.get_recent_builds(limit=limit, owner=owner, repo=repo)
                return [self._build_to_dict(b) for b in builds]
            
            elif operation == "get_failed_builds":
                limit = parameters.get('limit', 5)
                failed = self.github_service.get_failed_builds(limit=limit, owner=owner, repo=repo)
                return [self._build_to_dict(b) for b in failed]
            
            elif operation == "get_build_trends":
                days = parameters.get('days', 7)
                return self.github_service.get_build_trends(days=days, owner=owner, repo=repo)
            
            elif operation == "get_recent_commits":
                limit = parameters.get('limit', 20)
                branch = parameters.get('branch', 'main')
                return self.github_service.get_commit_history(
                    branch=branch,
                    limit=limit,
                    owner=owner,
                    repo=repo,
                )
            
            elif operation == "get_workflows":
                limit = parameters.get('limit', 10)
                workflow_id = parameters.get('workflow_id')
                return self.github_service.get_workflow_runs(
                    workflow_id=workflow_id,
                    limit=limit,
                    owner=owner,
                    repo=repo,
                )
            
            else:
                return {"error": f"Unknown GitHub operation: {operation}"}
                
        except Exception as e:
            logger.error(f"GitHub operation {operation} failed: {e}")
            return {"error": str(e)}
    
    def _build_to_dict(self, build) -> Dict[str, Any]:
        """Convert BuildData object to dictionary for JSON serialization"""
        if hasattr(build, '__dict__'):
            return {
                'build_id': getattr(build, 'build_id', None),
                'status': getattr(build, 'status', None),
                'start_time': str(getattr(build, 'start_time', None)),
                'duration_seconds': getattr(build, 'duration_seconds', None),
                'commit_sha': getattr(build, 'commit_sha', None),
                'branch': getattr(build, 'branch', None),
                'workflow_name': getattr(build, 'workflow_name', None),
            }
        return build

    @staticmethod
    def _extract_top_n_from_query(query: str, default: int = 10, minimum: int = 1, maximum: int = 100) -> int:
        """Extract requested list size from natural language like 'top 10 repos'."""
        text = (query or "").lower()
        match = re.search(r"(?:top|first|last)\s+(\d{1,3})", text)
        if not match:
            match = re.search(r"\b(\d{1,3})\s+(?:repo|repos|repositories)\b", text)
        if not match:
            return default
        try:
            value = int(match.group(1))
        except (TypeError, ValueError):
            return default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _is_my_github_repos_query(query: str) -> bool:
        """Detect explicit requests to list repositories belonging to the authenticated user."""
        text = (query or "").lower()
        has_github = "github" in text
        has_repo_term = any(term in text for term in [" repo", "repos", "repository", "repositories"])
        has_ownership = any(term in text for term in [" of mine", "my ", "mine", "i own", "owned by me"])
        has_list_intent = any(term in text for term in ["list", "show", "give", "fetch", "top"])
        return has_github and has_repo_term and has_ownership and has_list_intent

    def _build_my_repos_response(self, user_query: str) -> Optional[AgentResponse]:
        """Handle direct 'list my GitHub repos' prompts without relying on model tool selection."""
        if not self._is_my_github_repos_query(user_query):
            return None

        limit = self._extract_top_n_from_query(user_query, default=10, minimum=1, maximum=50)
        repos = self._execute_github_operation("get_accessible_repos", {"limit": limit})

        if isinstance(repos, dict) and repos.get("error"):
            message = repos.get("error", "GitHub query failed")
            content = f"I could not fetch your GitHub repositories: {message}"
        elif not isinstance(repos, list) or not repos:
            content = "I could not find any repositories accessible with the current GitHub token."
        else:
            ranked = sorted(
                repos,
                key=lambda item: (
                    item.get("stargazers_count", 0),
                    item.get("forks_count", 0),
                    item.get("updated_at") or ""
                ),
                reverse=True,
            )[:limit]

            lines = [f"Top {len(ranked)} GitHub repositories you can access:"]
            for idx, repo in enumerate(ranked, start=1):
                full_name = repo.get("full_name") or repo.get("name") or "unknown"
                stars = repo.get("stargazers_count", 0)
                forks = repo.get("forks_count", 0)
                language = repo.get("language") or "n/a"
                visibility = "private" if repo.get("private") else "public"
                lines.append(
                    f"{idx}. {full_name} ({visibility}) - ⭐ {stars} | forks {forks} | language {language}"
                )
            content = "\n".join(lines)

        return AgentResponse(
            agent_type=self.agent_type,
            task_id="github-repos-shortcut",
            status=TaskStatus.SUCCESS,
            insights=[
                Insight(
                    summary="GitHub repositories retrieved via deterministic repo-list handler",
                    severity=Severity.LOW,
                    business_impact="Improves reliability for repo listing prompts",
                    confidence_score=0.95,
                    recommendations=[
                        "Ask for repository-specific build, commit, or workflow analysis by mentioning owner/repo.",
                    ],
                )
            ],
            data={
                "full_response": content,
                "answer": content,
                "thinking": "",
                "tool_iterations": 1,
            },
            execution_time_ms=0.0,
        )


    @staticmethod
    def _parse_model_output(raw_content: str) -> Dict[str, str]:
        """
        Split model output into a user-safe answer and optional reasoning text.

        Some Bedrock models may emit internal protocol/tool text markers during tool use.
        This method extracts <reasoning> blocks for explicit UI display and removes
        transport markers from the final answer shown to end users.
        """
        text = (raw_content or "").strip()
        if not text:
            return {"answer": "", "thinking": ""}

        reasoning_blocks = [
            block.strip()
            for block in re.findall(r"<reasoning>([\s\S]*?)</reasoning>", text, flags=re.IGNORECASE)
            if block and block.strip()
        ]

        answer = re.sub(r"<reasoning>[\s\S]*?</reasoning>", "", text, flags=re.IGNORECASE).strip()

        # Remove transport tokens that should never be shown directly to users.
        answer = re.sub(r"<\|[^>]+\|>", "", answer)
        answer = re.sub(r"\s+to=functions\.[^\n\r]*", "", answer)

        # If protocol marker leaks inline, keep content before it.
        protocol_markers = ["<|start|>", "<|channel|>", "<|message|>"]
        for marker in protocol_markers:
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()

        # Normalize blank lines while preserving readable paragraph breaks.
        answer = re.sub(r"\n{3,}", "\n\n", answer).strip()

        return {
            "answer": answer,
            "thinking": "\n\n".join(reasoning_blocks).strip(),
        }
    
    def analyze_with_dynamic_tools(
        self,
        task_id: str,
        user_query: str,
        context: Dict[str, Any] = None
    ) -> AgentResponse:
        """
        Analyze observability data using dynamic AWS API tool calling.
        
        This method lets the LLM decide which AWS APIs to call based on the query,
        rather than having hardcoded API calls.
        
        Args:
            task_id: Task identifier
            user_query: Natural language query from user
            context: Additional context (e.g., AWS account info)
            
        Returns:
            AgentResponse with insights generated using dynamic tool execution
            
        Example queries the LLM can handle:
            - "Show me recent CloudWatch logs for errors"
            - "List all EC2 instances with high CPU"
            - "Get RDS database metrics for the last hour"
            - "Find Lambda functions with high error rates"
        """
        start_time = time.time()
        context = context or {}

        # Use per-user AWS credentials if provided in context
        creds = context.get('aws_credentials')
        if creds:
            self.bedrock_client = self.bedrock_client.with_credentials(
                access_key_id=creds['access_key_id'],
                secret_access_key=creds['secret_access_key'],
                session_token=creds.get('session_token'),
                region=creds.get('region'),
            )
            self.aws_executor = self.aws_executor.with_credentials(
                access_key_id=creds['access_key_id'],
                secret_access_key=creds['secret_access_key'],
                session_token=creds.get('session_token'),
                region=creds.get('region'),
            )

        try:
            direct_github_response = self._build_my_repos_response(user_query)
            if direct_github_response is not None:
                direct_github_response.task_id = task_id
                direct_github_response.execution_time_ms = (time.time() - start_time) * 1000
                return direct_github_response

            # Prepare system prompt for the LLM
            system_prompt = """You are an expert AWS observability agent analyzing infrastructure and CI/CD pipelines.

Your goal: Answer the user's query by making appropriate AWS API calls and/or GitHub repository queries.

Available tools:
1. aws_api_executor - Call ANY AWS service (ec2, cloudwatch, logs, rds, lambda, etc.)
2. github_query - Query GitHub repository data (builds, commits, workflows, CI/CD metrics)

AWS Guidelines:
- For CloudWatch metrics: Use 'cloudwatch' service with operations like 'list_metrics', 'get_metric_statistics'
- For logs: Use 'logs' service with operations like 'describe_log_groups', 'filter_log_events'
- For EC2: Use 'ec2' service with operations like 'describe_instances'
- For RDS: Use 'rds' service with operations like 'describe_db_instances'
- CRITICAL: Do NOT use 'MaxRecords' parameter with CloudWatch list_metrics - it's not supported!

GitHub Guidelines:
- Use github_query when user asks about repository, builds, CI/CD, workflows, pull requests, or commits
- Target any repository by passing owner/repo (or repository='owner/repo') in github_query parameters
- Available operations: get_repo_info, get_recent_builds, get_failed_builds, get_build_trends, get_recent_commits, get_workflows, get_accessible_repos

After gathering data, provide insights in natural language."""
            
            # Add AWS account context if available
            if context.get('aws_account'):
                acc = context['aws_account']
                system_prompt += f"\n\nCurrent AWS Context:\n- Account: {acc.get('account_id')}\n- Region: {acc.get('region')}\n- Monthly Cost: ${acc.get('monthly_unblended_cost_usd', 0):.2f}"
            
            # Add GitHub context if configured
            if self.github_service.is_configured():
                system_prompt += (
                    "\n\nGitHub token is configured. "
                    "Default repository may be set in .env, but github_query can target any owner/repo explicitly."
                )
            
            user_prompt = f"User Query: {user_query}\n\nPlease analyze this query and provide insights."
            
            # Get tool definitions
            tools = [self.aws_executor.get_tool_definition()]
            
            # Add GitHub tool if configured
            if self.github_service.is_configured():
                tools.append(self._get_github_tool_definition())
            
            # Define tool executor function
            def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> Any:
                """Execute AWS API calls or GitHub queries requested by the LLM"""
                if tool_name == "aws_api_executor":
                    try:
                        result = self.aws_executor.execute(
                            service=tool_input['service'],
                            operation=tool_input['operation'],
                            parameters=tool_input.get('parameters', {})
                        )
                        logger.info(f"Tool executed: {tool_input['service']}.{tool_input['operation']}")
                        return result
                    except Exception as e:
                        logger.error(f"Tool execution failed: {e}")
                        return {"error": str(e)}
                
                elif tool_name == "github_query":
                    try:
                        result = self._execute_github_operation(
                            operation=tool_input['operation'],
                            parameters=tool_input.get('parameters', {})
                        )
                        logger.info(f"GitHub tool executed: {tool_input['operation']}")
                        return result
                    except Exception as e:
                        logger.error(f"GitHub tool execution failed: {e}")
                        return {"error": str(e)}
                
                else:
                    return {"error": f"Unknown tool: {tool_name}"}
            
            # Invoke Claude with tool support
            response = self.bedrock_client.invoke_with_tools(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                tools=tools,
                tool_executor=execute_tool,
                max_iterations=10,
                max_output_tokens=1200,
                temperature=0.0,
                use_haiku=False  # Use Sonnet for better reasoning
            )

            parsed_output = self._parse_model_output(response.get('content', ''))
            display_answer = parsed_output.get('answer') or response.get('content', '')
            thinking = parsed_output.get('thinking', '')
            
            # Ensure display_answer is never empty
            if not display_answer or not display_answer.strip():
                display_answer = "Analysis completed. Please check the AWS console for detailed metrics and logs."
            
            # Extract insights from response - create brief summary instead of full response
            tool_count = response.get('iterations', 0)
            insights = []
            
            if tool_count > 0:
                insights.append(Insight(
                    summary=f"Observability analysis completed using {tool_count} AWS API call(s)",
                    severity=Severity.LOW,
                    business_impact="Live AWS data retrieved and analyzed",
                    confidence_score=0.9,
                    recommendations=[
                        "Review the detailed analysis in the response"
                    ]
                ))
            
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                insights=insights,
                data={
                    'full_response': display_answer,
                    'raw_model_response': response.get('content', ''),
                    'thinking': thinking,
                    'tool_iterations': response.get('iterations', 0),
                    'tokens_used': response.get('usage', {}),
                    'answer': display_answer  # Make sure answer is available for planner
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Dynamic tool analysis failed: {e}")

            # Fallback path: still answer the user even if tool-calling is unavailable.
            try:
                fallback_system_prompt = """You are an AWS observability expert.

Tool-calling is unavailable in this environment. Provide best-effort guidance based on:
- User query
- AWS account context
- Observability best practices

Be explicit when an answer is guidance vs verified live data."""

                fallback_prompt = f"""User Query: {user_query}

AWS Context: {context.get('aws_account', {})}

Provide actionable observability guidance and next checks."""

                fallback = self.bedrock_client.invoke_claude_sonnet(
                    system_prompt=fallback_system_prompt,
                    user_prompt=fallback_prompt,
                    temperature=0.0,
                    max_tokens=1024,
                )

                fallback_text = (fallback.get('content') or '').strip()
                if not fallback_text:
                    fallback_text = (
                        "I could not fetch live observability data from AWS tools in this run. "
                        "Please verify CloudWatch metric/log permissions and Bedrock model tool-call compatibility."
                    )

                execution_time = (time.time() - start_time) * 1000
                return AgentResponse(
                    agent_type=self.agent_type,
                    task_id=task_id,
                    status=TaskStatus.PARTIAL,
                    insights=[],
                    data={
                        'full_response': fallback_text,
                        'tool_error': str(e),
                    },
                    execution_time_ms=execution_time,
                )
            except Exception as fallback_error:
                logger.error(f"Observability fallback also failed: {fallback_error}")
                execution_time = (time.time() - start_time) * 1000

                return AgentResponse(
                    agent_type=self.agent_type,
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    insights=[],
                    data={'error': str(e), 'fallback_error': str(fallback_error)},
                    execution_time_ms=execution_time,
                    error_message=str(e)
                )
    
    def analyze_metrics(
        self,
        task_id: str,
        metrics: List[MetricData],
        context: Dict[str, Any] = None
    ) -> AgentResponse:
        """
        Analyze metrics and generate semantic insights.
        
        Args:
            task_id: Task identifier
            metrics: List of metric data points
            context: Additional context
            
        Returns:
            AgentResponse with insights
        """
        start_time = time.time()
        context = context or {}
        
        try:
            # Retrieve historical context
            historical_context = ""
            if self.knowledge_base:
                metric_names = list(set(m.metric_name for m in metrics))
                for metric_name in metric_names[:3]:  # Limit context retrieval
                    docs = self.knowledge_base.query_context(
                        query=f"historical {metric_name} patterns and baselines",
                        max_results=2
                    )
                    if docs:
                        historical_context += f"\n{metric_name} context: {docs[0]['content'][:200]}"
            
            # Detect anomalies
            anomalies = self.detect_anomalies(metrics)
            
            # Generate semantic insights using Claude
            insights = self._generate_insights(metrics, anomalies, historical_context, context)
            
            # Store metrics in knowledge base for future reference
            if self.knowledge_base:
                self.knowledge_base.store_metrics(metrics)
            
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                insights=insights,
                data={
                    'metrics_analyzed': len(metrics),
                    'anomalies_detected': len(anomalies),
                    'anomalies': [
                        {
                            'metric': a.metric_name,
                            'expected': a.expected_value,
                            'observed': a.observed_value,
                            'deviation': a.deviation_sigma,
                        }
                        for a in anomalies
                    ],
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Observability Agent failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.FAILED,
                insights=[],
                data={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def detect_anomalies(
        self,
        metrics: List[MetricData],
        sigma_threshold: float = 2.0
    ) -> List[Anomaly]:
        """
        Detect anomalies using statistical analysis (2-sigma deviation).
        
        Args:
            metrics: Metric data points
            sigma_threshold: Standard deviation threshold
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        # Group metrics by name
        metric_groups = {}
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric)
        
        # Detect anomalies in each group
        for metric_name, metric_list in metric_groups.items():
            if len(metric_list) < 3:
                continue  # Need at least 3 points for statistics
            
            values = [m.value for m in metric_list]
            mean_value = statistics.mean(values)
            
            try:
                stdev = statistics.stdev(values)
            except statistics.StatisticsError:
                continue  # All values are the same
            
            if stdev == 0:
                continue
            
            # Check each value for anomalies
            for metric in metric_list:
                deviation = abs(metric.value - mean_value) / stdev
                
                if deviation >= sigma_threshold:
                    # Calculate confidence based on deviation
                    confidence = min(0.95, 0.5 + (deviation / (sigma_threshold * 2)))
                    
                    anomaly = Anomaly(
                        metric_name=metric.metric_name,
                        expected_value=mean_value,
                        observed_value=metric.value,
                        deviation_sigma=deviation,
                        confidence_score=confidence,
                        timestamp=metric.timestamp,
                        context=f"{metric.metric_type.value} spike detected"
                    )
                    anomalies.append(anomaly)
                    
                    logger.info(
                        f"Anomaly detected: {metric_name} = {metric.value:.2f} "
                        f"(expected {mean_value:.2f}, {deviation:.1f}σ)"
                    )
        
        return anomalies
    
    def attribute_bottleneck(
        self,
        task_id: str,
        performance_issue: str,
        metrics: List[MetricData]
    ) -> AgentResponse:
        """
        Attribute performance bottleneck to infrastructure components.
        
        Args:
            task_id: Task ID
            performance_issue: Description of performance issue
            metrics: Related metrics
            
        Returns:
            AgentResponse with attribution insights
        """
        start_time = time.time()
        
        try:
            # Build context from metrics
            metric_summary = "\n".join([
                f"- {m.metric_name}: {m.value} {m.unit} at {m.timestamp}"
                for m in metrics[:10]
            ])
            
            # Query knowledge base for similar issues
            similar_issues = []
            if self.knowledge_base:
                similar_issues = self.knowledge_base.query_context(
                    query=f"performance bottleneck {performance_issue}",
                    max_results=3
                )
            
            # Generate attribution using Claude
            system_prompt = """You are an expert SRE analyzing infrastructure performance issues.
Your task is to attribute performance bottlenecks to specific infrastructure components.

Analyze the metrics and provide:
1. Root cause component (e.g., Redis, Postgres, ECS task, network)
2. Confidence score (0.0-1.0)
3. Recommended actions

Return JSON:
{
  "root_cause_component": "string",
  "confidence_score": 0.0,
  "reasoning": "string",
  "recommendations": ["action1", "action2"]
}"""
            
            user_prompt = f"""Performance Issue: {performance_issue}

Metrics:
{metric_summary}

Similar Historical Issues:
{json.dumps([s.get('content', '')[:200] for s in similar_issues[:2]])}

Attribute this bottleneck to a specific component."""
            
            import json
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Create insight from attribution
            insight = Insight(
                summary=f"Bottleneck attributed to: {response.get('root_cause_component', 'Unknown')}",
                business_impact=f"Performance issue: {performance_issue}",
                severity=Severity.HIGH,
                recommendations=response.get('recommendations', []),
                confidence_score=response.get('confidence_score', 0.5)
            )
            
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                insights=[insight],
                data={
                    'attribution': response,
                    'metrics_analyzed': len(metrics),
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Bottleneck attribution failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.FAILED,
                insights=[],
                data={},
                execution_time_ms=execution_time,
                error_message=str(e)
            )
    
    def _generate_insights(
        self,
        metrics: List[MetricData],
        anomalies: List[Anomaly],
        historical_context: str,
        context: Dict[str, Any]
    ) -> List[Insight]:
        """Generate semantic insights from metrics and anomalies"""
        
        # Build metric summary
        metric_summary = self._summarize_metrics(metrics)
        anomaly_summary = self._summarize_anomalies(anomalies)
        
        system_prompt = """You are an expert SRE providing semantic insights from infrastructure metrics.

Your goal: Transform raw metrics into actionable business insights.

Example:
Input: CPU: 78%, Memory: 85%, Redis Memory: 12GB
Output: "Your Celery worker pool is mis-sized relative to Redis throughput causing job starvation. Recommend worker pool = 12. Projected cost reduction = ₹18,300/month."

Provide insights as JSON array:
[
  {
    "summary": "brief insight (1 sentence)",
    "business_impact": "business impact explanation",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["action1", "action2"],
    "cost_impact_inr": 0.0,
    "confidence_score": 0.0
  }
]"""
        
        user_prompt = f"""Metrics:
{metric_summary}

Anomalies Detected:
{anomaly_summary}

Historical Context:
{historical_context}

Additional Context:
{context.get('description', 'General metric analysis')}

Generate 1-3 semantic insights with business impact."""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            # Convert to Insight objects
            insights = []
            insight_list = response if isinstance(response, list) else [response]
            
            for insight_data in insight_list:
                try:
                    insight = Insight(
                        summary=insight_data.get('summary', 'No summary'),
                        business_impact=insight_data.get('business_impact', ''),
                        severity=Severity(insight_data.get('severity', 'MEDIUM')),
                        recommendations=insight_data.get('recommendations', []),
                        cost_impact_inr=float(insight_data.get('cost_impact_inr', 0.0)),
                        confidence_score=float(insight_data.get('confidence_score', 0.7))
                    )
                    insights.append(insight)
                except Exception as e:
                    logger.error(f"Failed to create insight: {e}")
            
            return insights
            
        except Exception as e:
            logger.error(f"Failed to generate insights: {e}")
            # Return fallback insight
            return [
                Insight(
                    summary=f"Analyzed {len(metrics)} metrics, detected {len(anomalies)} anomalies",
                    business_impact="Metrics analyzed successfully",
                    severity=Severity.LOW,
                    recommendations=["Review metric trends", "Monitor for patterns"],
                    confidence_score=0.5
                )
            ]
    
    def _summarize_metrics(self, metrics: List[MetricData]) -> str:
        """Summarize metrics for LLM prompt"""
        if not metrics:
            return "No metrics available"
        
        summary_lines = []
        metric_groups = {}
        
        for metric in metrics:
            if metric.metric_name not in metric_groups:
                metric_groups[metric.metric_name] = []
            metric_groups[metric.metric_name].append(metric.value)
        
        for name, values in metric_groups.items():
            avg_value = statistics.mean(values)
            max_value = max(values)
            min_value = min(values)
            
            summary_lines.append(
                f"- {name}: avg={avg_value:.2f}, max={max_value:.2f}, min={min_value:.2f} ({len(values)} points)"
            )
        
        return "\n".join(summary_lines[:15])  # Limit to 15 metric summaries
    
    def _summarize_anomalies(self, anomalies: List[Anomaly]) -> str:
        """Summarize anomalies for LLM prompt"""
        if not anomalies:
            return "No anomalies detected"
        
        summary_lines = []
        for anomaly in anomalies[:10]:  # Limit to 10
            summary_lines.append(
                f"- {anomaly.metric_name}: expected {anomaly.expected_value:.2f}, "
                f"observed {anomaly.observed_value:.2f} "
                f"({anomaly.deviation_sigma:.1f}σ deviation, "
                f"{anomaly.confidence_score:.0%} confidence)"
            )
        
        return "\n".join(summary_lines)
