"""
Database Agent

Responsible for:
- Query plan analysis
- Index recommendations
- Database performance optimization
- Schema analysis
"""

import logging
import time
from typing import Dict, List, Optional, Any

from models.agent_protocol import AgentResponse, AgentType, TaskStatus, Severity, Insight
from models.core_models import QueryPattern, Recommendation
from services.bedrock_client import BedrockClient
from services.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


class DBAgent:
    """Database optimization agent"""
    
    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        knowledge_base: KnowledgeBase = None,
    ):
        """Initialize Database Agent"""
        self.bedrock_client = bedrock_client or BedrockClient()
        self.knowledge_base = knowledge_base
        self.agent_type = AgentType.DB
        
        logger.info("Database Agent initialized")
    
    def process_task(
        self,
        task_id: str,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Process database analysis task"""
        start_time = time.time()
        
        try:
            # Check if query pattern is provided
            query_pattern = context.get('query_pattern')
            
            if query_pattern:
                insights = self._analyze_query_pattern(query_pattern, context)
            else:
                # General database query
                insights = self._analyze_database_query(query, context)
            
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                insights=insights,
                data={'analysis_type': 'query_pattern' if query_pattern else 'query'},
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Database Agent failed: {e}")
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
    
    def _analyze_query_pattern(
        self,
        pattern: QueryPattern,
        context: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze database query pattern"""
        
        system_prompt = """You are a database performance expert analyzing query patterns.

Provide:
1. Index recommendations with CREATE INDEX statements
2. Query optimization suggestions
3. Expected performance improvement
4. Cost impact if relevant

Return JSON array:
[
  {
    "summary": "optimization insight",
    "business_impact": "performance impact",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["CREATE INDEX ...", "action2"],
    "confidence_score": 0.0
  }
]"""
        
        user_prompt = f"""Query Pattern:
Template: {pattern.query_template}
Execution Count: {pattern.execution_count}
Avg Duration: {pattern.avg_duration_ms}ms
Max Duration: {pattern.max_duration_ms}ms
Tables: {', '.join(pattern.tables_accessed)}
Missing Indices: {', '.join(pattern.missing_indices) if pattern.missing_indices else 'None detected'}

Analyze and recommend optimizations."""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            insights = []
            insight_list = response if isinstance(response, list) else [response]
            
            for insight_data in insight_list:
                insight = Insight(
                    summary=insight_data.get('summary', 'Query optimization available'),
                    business_impact=insight_data.get('business_impact', ''),
                    severity=Severity(insight_data.get('severity', 'MEDIUM')),
                    recommendations=insight_data.get('recommendations', []),
                    confidence_score=float(insight_data.get('confidence_score', 0.8))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Query pattern analysis failed: {e}")
            return []
    
    def _analyze_database_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Insight]:
        """Analyze general database query"""
        
        system_prompt = """You are a database expert providing optimization guidance.

Return JSON array:
[
  {
    "summary": "database recommendation",
    "business_impact": "impact",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["action1"],
    "confidence_score": 0.0
  }
]"""
        
        user_prompt = f"""Query: {query}

Context: {context.get('description', '')}

Provide database optimization recommendations."""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            insight_list = response if isinstance(response, list) else [response]
            
            insights = []
            for insight_data in insight_list:
                insight = Insight(
                    summary=insight_data.get('summary', 'Database guidance'),
                    business_impact=insight_data.get('business_impact', ''),
                    severity=Severity(insight_data.get('severity', 'MEDIUM')),
                    recommendations=insight_data.get('recommendations', []),
                    confidence_score=float(insight_data.get('confidence_score', 0.7))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Database query analysis failed: {e}")
            return []
