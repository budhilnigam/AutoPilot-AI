"""
CI/CD Agent

Responsible for:
- Build time analysis and regression detection
- GitHub Actions workflow monitoring
- Deployment tracking
- Build optimization recommendations
"""

import logging
import time
from typing import Dict, List, Optional, Any

from models.agent_protocol import AgentResponse, AgentType, TaskStatus, Severity, Insight
from models.core_models import BuildData
from services.bedrock_client import BedrockClient
from services.knowledge_base import KnowledgeBase
from services.github_client import GitHubClient
from services.github_service import GitHubService
from config import config

logger = logging.getLogger(__name__)


class CICDAgent:
    """CI/CD performance monitoring and optimization agent"""
    
    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        knowledge_base: KnowledgeBase = None,
        github_client: GitHubClient = None,
    ):
        """Initialize CI/CD Agent"""
        self.bedrock_client = bedrock_client or BedrockClient()
        self.knowledge_base = knowledge_base
        self.github_client = github_client or GitHubClient()
        self.github_service = GitHubService(token=config.GITHUB_TOKEN)
        self.agent_type = AgentType.CICD
        
        logger.info("CI/CD Agent initialized")
    
    def process_task(
        self,
        task_id: str,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Process CI/CD analysis task"""
        start_time = time.time()
        
        try:
            # Check if build data is provided
            build_data = context.get('build_data')
            repository = context.get('repository')
            
            if build_data:
                insights = self._analyze_build_data(build_data, context)
            elif repository:
                insights = self._analyze_repository_builds(repository, context)
            else:
                insights = self._analyze_cicd_query(query, context)
            
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                insights=insights,
                data={'analysis_type': 'build_data' if build_data else 'repository' if repository else 'query'},
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"CI/CD Agent failed: {e}")
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
    
    def _analyze_build_data(
        self,
        build_data: BuildData,
        context: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze build performance data"""
        
        system_prompt = """You are a CI/CD optimization expert analyzing build performance.

Provide:
1. Build time regression analysis
2. Optimization opportunities (caching, parallelization)
3. Bottleneck identification
4. Recommendations with expected time savings

Return JSON array:
[
  {
    "summary": "build optimization insight",
    "business_impact": "impact on development velocity",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["action1", "action2"],
    "confidence_score": 0.0
  }
]"""
        
        steps_summary = "\n".join([
            f"- {step.get('name', 'Unknown')}: {step.get('status', 'unknown')} ({step.get('conclusion', 'N/A')})"
            for step in build_data.steps[:15]
        ])
        
        user_prompt = f"""Build Data:
Build ID: {build_data.build_id}
Commit: {build_data.commit_sha}
Build Time: {build_data.build_time_seconds:.1f} seconds
Status: {build_data.status}
Repository: {build_data.repository}
Branch: {build_data.branch}

Build Steps:
{steps_summary}

Analyze build performance and recommend optimizations."""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            insights = []
            insight_list = response if isinstance(response, list) else [response]
            
            for insight_data in insight_list:
                insight = Insight(
                    summary=insight_data.get('summary', 'Build analysis completed'),
                    business_impact=insight_data.get('business_impact', ''),
                    severity=Severity(insight_data.get('severity', 'MEDIUM')),
                    recommendations=insight_data.get('recommendations', []),
                    confidence_score=float(insight_data.get('confidence_score', 0.7))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Build data analysis failed: {e}")
            return []
    
    def _analyze_repository_builds(
        self,
        repository: str,
        context: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze build trends for a repository"""
        
        try:
            # Parse owner/repo
            parts = repository.split('/')
            if len(parts) != 2:
                raise ValueError("Repository must be in format owner/repo")
            
            owner, repo = parts
            
            # Get build trends
            trends = self.github_client.analyze_build_trends(
                owner=owner,
                repo=repo,
                limit=30
            )
            
            system_prompt = """You are a CI/CD expert analyzing build trends.

Identify:
1. Build time regressions
2. Failure patterns
3. Optimization opportunities

Return JSON array:
[
  {
    "summary": "trend insight",
    "business_impact": "development impact",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["action1"],
    "confidence_score": 0.0
  }
]"""
            
            user_prompt = f"""Repository: {repository}

Build Trends:
- Total Runs: {trends['total_runs']}
- Success Rate: {trends['success_rate']:.1%}
- Avg Build Time: {trends['avg_build_time_seconds']:.1f}s
- Max Build Time: {trends['max_build_time_seconds']:.1f}s
- Min Build Time: {trends['min_build_time_seconds']:.1f}s

Recent Build Times: {trends['build_times'][:10]}

Analyze trends and recommend improvements."""
            
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            insights = []
            insight_list = response if isinstance(response, list) else [response]
            
            for insight_data in insight_list:
                insight = Insight(
                    summary=insight_data.get('summary', 'Build trends analyzed'),
                    business_impact=insight_data.get('business_impact', ''),
                    severity=Severity(insight_data.get('severity', 'MEDIUM')),
                    recommendations=insight_data.get('recommendations', []),
                    confidence_score=float(insight_data.get('confidence_score', 0.7))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Repository build analysis failed: {e}")
            return []
    
    def _analyze_cicd_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze general CI/CD query"""
        
        system_prompt = """You are a CI/CD expert providing optimization guidance.

Return JSON array:
[
  {
    "summary": "CI/CD recommendation",
    "business_impact": "impact",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["action1"],
    "confidence_score": 0.0
  }
]"""
        
        user_prompt = f"""Query: {query}

Context: {context.get('description', '')}

Provide CI/CD optimization recommendations."""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            insight_list = response if isinstance(response, list) else [response]
            
            insights = []
            for insight_data in insight_list:
                insight = Insight(
                    summary=insight_data.get('summary', 'CI/CD guidance'),
                    business_impact=insight_data.get('business_impact', ''),
                    severity=Severity(insight_data.get('severity', 'MEDIUM')),
                    recommendations=insight_data.get('recommendations', []),
                    confidence_score=float(insight_data.get('confidence_score', 0.7))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"CI/CD query analysis failed: {e}")
            return []
    
    def analyze_github_repository_health(self) -> Dict[str, Any]:
        """
        Analyze GitHub repository build health.
        
        Returns:
            Health analysis with insights
        """
        if not self.github_service.is_configured():
            logger.warning("GitHub not configured for health analysis")
            return {
                'status': 'unconfigured',
                'message': 'GitHub service not configured. Set GITHUB_TOKEN, GITHUB_REPO_OWNER, and GITHUB_REPO_NAME.'
            }
        
        try:
            health = self.github_service.get_build_health_summary()
            failed_builds = self.github_service.get_failed_builds(limit=3)
            
            insights = []
            if health.get('status') == 'success':
                # Create insights based on health data
                success_rate = health.get('success_rate_percent', 0)
                
                # Build time insight
                avg_build_time = health.get('average_build_time_seconds', 0)
                
                summary = f"Build Health: {health['health'].upper()} - {success_rate:.1f}% success rate"
                
                recommendations = []
                if success_rate < 90:
                    recommendations.append(f"Investigate failing builds: {len(failed_builds)} failures in recent runs")
                if avg_build_time > 600:  # 10 minutes
                    recommendations.append("Build time is high. Consider parallelization or caching optimization")
                
                insight = Insight(
                    summary=summary,
                    business_impact=f"Development velocity at {success_rate:.1f}%",
                    severity=Severity.CRITICAL if success_rate < 70 else Severity.HIGH if success_rate < 90 else Severity.LOW,
                    recommendations=recommendations,
                    confidence_score=0.95
                )
                insights.append(insight)
            
            return {
                'status': 'success',
                'health': health,
                'insights': [
                    {
                        'summary': i.summary,
                        'severity': i.severity.value,
                        'recommendations': i.recommendations
                    }
                    for i in insights
                ]
            }
            
        except Exception as e:
            logger.error(f"GitHub health analysis failed: {e}")
            return {'status': 'error', 'message': str(e)}
