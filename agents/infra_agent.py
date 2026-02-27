"""
Infrastructure Agent

Responsible for:
- Docker and ECS configuration analysis
- Infrastructure drift detection
- Configuration optimization
- Resource right-sizing
"""

import logging
import time
from typing import Dict, List, Optional, Any

from models.agent_protocol import AgentResponse, AgentType, TaskStatus, Severity, Insight
from models.core_models import Configuration, Recommendation
from services.bedrock_client import BedrockClient
from services.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class InfraAgent:
    """Infrastructure configuration analysis agent"""
    
    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        knowledge_base: KnowledgeBase = None,
    ):
        """Initialize Infrastructure Agent"""
        self.bedrock_client = bedrock_client or BedrockClient()
        self.knowledge_base = knowledge_base
        self.agent_type = AgentType.INFRA
        
        logger.info("Infrastructure Agent initialized")
    
    def process_task(
        self,
        task_id: str,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Process infrastructure analysis task"""
        start_time = time.time()
        
        try:
            # Check if configuration is provided
            config = context.get('configuration')
            
            if config:
                insights = self._analyze_configuration(config, context)
            else:
                # General infrastructure query
                insights = self._analyze_infrastructure_query(query, context)
            
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                insights=insights,
                data={'analysis_type': 'configuration' if config else 'query'},
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Infrastructure Agent failed: {e}")
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
    
    def _analyze_configuration(
        self,
        config: Configuration,
        context: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze infrastructure configuration"""
        
        # Retrieve baseline configuration
        baseline_context = ""
        if self.knowledge_base:
            docs = self.knowledge_base.query_context(
                query=f"baseline {config.config_type} configuration best practices",
                max_results=2
            )
            if docs:
                baseline_context = docs[0]['content'][:300]
        
        system_prompt = """You are an expert DevOps engineer analyzing infrastructure configurations.

Analyze the configuration and provide:
1. Optimization opportunities
2. Security issues
3. Performance improvements
4. Cost reduction possibilities

Return JSON array of insights:
[
  {
    "summary": "brief insight",
    "business_impact": "impact explanation",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["action1", "action2"],
    "confidence_score": 0.0
  }
]"""
        
        user_prompt = f"""Configuration Type: {config.config_type}

Configuration:
```
{config.config_content[:1000]}
```

Baseline Best Practices:
{baseline_context}

Metadata: {config.metadata}

Analyze this configuration."""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            insights = []
            insight_list = response if isinstance(response, list) else [response]
            
            for insight_data in insight_list:
                insight = Insight(
                    summary=insight_data.get('summary', 'Configuration analyzed'),
                    business_impact=insight_data.get('business_impact', ''),
                    severity=Severity(insight_data.get('severity', 'MEDIUM')),
                    recommendations=insight_data.get('recommendations', []),
                    confidence_score=float(insight_data.get('confidence_score', 0.7))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Configuration analysis failed: {e}")
            return [
                Insight(
                    summary="Configuration analysis completed",
                    business_impact="Unable to generate detailed insights",
                    severity=Severity.LOW,
                    recommendations=["Review configuration manually"],
                    confidence_score=0.3
                )
            ]
    
    def _analyze_infrastructure_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze general infrastructure query"""
        
        system_prompt = """You are an expert DevOps consultant providing infrastructure guidance.

Provide actionable recommendations for infrastructure questions.

Return JSON array:
[
  {
    "summary": "recommendation",
    "business_impact": "impact",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["action1"],
    "confidence_score": 0.0
  }
]"""
        
        user_prompt = f"""Query: {query}

Context: {context.get('description', '')}

Provide infrastructure recommendations."""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            insight_list = response if isinstance(response, list) else [response]
            
            insights = []
            for insight_data in insight_list:
                insight = Insight(
                    summary=insight_data.get('summary', 'Infrastructure guidance'),
                    business_impact=insight_data.get('business_impact', ''),
                    severity=Severity(insight_data.get('severity', 'MEDIUM')),
                    recommendations=insight_data.get('recommendations', []),
                    confidence_score=float(insight_data.get('confidence_score', 0.7))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Infrastructure query analysis failed: {e}")
            return []
