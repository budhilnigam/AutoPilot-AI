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
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

from models.agent_protocol import AgentResponse, AgentType, TaskStatus, Severity, Insight
from models.core_models import MetricData, Anomaly, Recommendation
from services.bedrock_client import BedrockClient
from services.knowledge_base import KnowledgeBase
from services.cloudwatch_client import CloudWatchClient

logger = logging.getLogger(__name__)


class ObservabilityAgent:
    """
    Observability Agent for metric analysis and anomaly detection.
    
    Generates semantic insights with business context.
    """
    
    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        knowledge_base: KnowledgeBase = None,
        cloudwatch_client: CloudWatchClient = None,
    ):
        """
        Initialize Observability Agent.
        
        Args:
            bedrock_client: Bedrock client for LLM
            knowledge_base: Knowledge base for RAG
            cloudwatch_client: CloudWatch client for metrics
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.knowledge_base = knowledge_base
        self.cloudwatch_client = cloudwatch_client or CloudWatchClient()
        self.agent_type = AgentType.OBSERVABILITY
        
        logger.info("Observability Agent initialized")
    
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
