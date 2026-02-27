"""
Cost Agent

Responsible for:
- Infrastructure cost optimization
- Pricing analysis with INR conversion
- Right-sizing recommendations
- Cost projections and savings calculations
"""

import logging
import time
from typing import Dict, List, Optional, Any

from models.agent_protocol import AgentResponse, AgentType, TaskStatus, Severity, Insight
from models.core_models import CostImpact, MetricData
from services.bedrock_client import BedrockClient
from services.knowledge_base import KnowledgeBase
from services.billing_client import BillingClient

logger = logging.getLogger(__name__)


class CostAgent:
    """Cost optimization and pricing analysis agent"""
    
    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        knowledge_base: KnowledgeBase = None,
        billing_client: BillingClient = None,
    ):
        """Initialize Cost Agent"""
        self.bedrock_client = bedrock_client or BedrockClient()
        self.knowledge_base = knowledge_base
        self.billing_client = billing_client or BillingClient()
        self.agent_type = AgentType.COST
        
        logger.info("Cost Agent initialized")
    
    def process_task(
        self,
        task_id: str,
        query: str,
        context: Dict[str, Any]
    ) -> AgentResponse:
        """Process cost analysis task"""
        start_time = time.time()
        
        try:
            # Get current cost data
            monthly_cost_inr = self.billing_client.get_monthly_cost_inr()
            
            # Analyze for cost optimization
            insights = self._analyze_cost_optimization(query, context, monthly_cost_inr)
            
            execution_time = (time.time() - start_time) * 1000
            
            return AgentResponse(
                agent_type=self.agent_type,
                task_id=task_id,
                status=TaskStatus.SUCCESS,
                insights=insights,
                data={
                    'current_monthly_cost_inr': monthly_cost_inr,
                    'currency': 'INR'
                },
                execution_time_ms=execution_time
            )
            
        except Exception as e:
            logger.error(f"Cost Agent failed: {e}")
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
    
    def _analyze_cost_optimization(
        self,
        query: str,
        context: Dict[str, Any],
        current_cost_inr: float
    ) -> List[Insight]:
        """Analyze cost optimization opportunities"""
        
        # Get service breakdown
        try:
            from datetime import datetime, timedelta
            end_date = datetime.utcnow().date()
            start_date = end_date - timedelta(days=30)
            
            service_costs = self.billing_client.get_service_costs_inr(
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
        except Exception as e:
            logger.warning(f"Could not get service costs: {e}")
            service_costs = {}
        
        # Retrieve cost context
        cost_context = ""
        if self.knowledge_base:
            docs = self.knowledge_base.query_context(
                query="AWS cost optimization best practices right-sizing",
                max_results=2
            )
            if docs:
                cost_context = docs[0]['content'][:300]
        
        system_prompt = """You are an AWS cost optimization expert for Indian startups.

Analyze costs and provide:
1. Right-sizing opportunities
2. Reserved instance recommendations
3. Savings plan suggestions
4. Cost-performance tradeoffs
5. Projected monthly savings in INR
6. Projected annual savings in INR

CRITICAL: All costs must be in INR (₹).

Return JSON array:
[
  {
    "summary": "cost optimization opportunity",
    "business_impact": "impact explanation",
    "severity": "LOW|MEDIUM|HIGH|CRITICAL",
    "recommendations": ["action1 with cost in INR", "action2"],
    "cost_impact_inr": -15000.0,
    "confidence_score": 0.0
  }
]

Negative cost_impact_inr = savings, Positive = additional cost"""
        
        metrics_summary = ""
        if 'metrics' in context:
            metrics = context['metrics']
            metrics_summary = "\n".join([
                f"- {m.metric_name}: {m.value} {m.unit}"
                for m in metrics[:10]
            ])
        
        user_prompt = f"""Current Monthly Cost: ₹{current_cost_inr:.2f} INR

Service Cost Breakdown:
{self._format_service_costs(service_costs)}

Metrics:
{metrics_summary}

Query: {query}

Best Practices Context:
{cost_context}

Analyze costs and recommend optimizations. All amounts must be in INR."""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt
            )
            
            insights = []
            insight_list = response if isinstance(response, list) else [response]
            
            for insight_data in insight_list:
                cost_impact = float(insight_data.get('cost_impact_inr', 0.0))
                
                insight = Insight(
                    summary=insight_data.get('summary', 'Cost optimization identified'),
                    business_impact=insight_data.get('business_impact', ''),
                    severity=Severity(insight_data.get('severity', 'MEDIUM')),
                    recommendations=insight_data.get('recommendations', []),
                    cost_impact_inr=cost_impact,
                    confidence_score=float(insight_data.get('confidence_score', 0.7))
                )
                insights.append(insight)
            
            return insights
            
        except Exception as e:
            logger.error(f"Cost optimization analysis failed: {e}")
            return [
                Insight(
                    summary=f"Current monthly cost: ₹{current_cost_inr:.2f}",
                    business_impact="Cost analysis completed",
                    severity=Severity.LOW,
                    recommendations=["Review AWS Cost Explorer for detailed breakdown"],
                    cost_impact_inr=0.0,
                    confidence_score=0.5
                )
            ]
    
    def _format_service_costs(self, service_costs: Dict[str, float]) -> str:
        """Format service costs for LLM prompt"""
        if not service_costs:
            return "No service breakdown available"
        
        # Sort by cost descending
        sorted_services = sorted(
            service_costs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        lines = []
        for service, cost_inr in sorted_services[:10]:  # Top 10
            lines.append(f"- {service}: ₹{cost_inr:.2f} INR")
        
        return "\n".join(lines)
