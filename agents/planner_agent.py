"""
Planner Agent

Central orchestrator for the multi-agent system.

Responsibilities:
- Analyze user queries/tasks
- Determine which specialized agents to invoke
- Route tasks to appropriate agents
- Synthesize responses from multiple agents
- Coordinate execution sequence
"""

import logging
import time
import uuid
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from models.agent_protocol import AgentResponse, AgentType, TaskStatus, Insight, Severity
from models.core_models import Task
from services.bedrock_client import BedrockClient
from services.knowledge_base import KnowledgeBase

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """Describes what a specialized agent can do"""
    agent_type: AgentType
    capabilities: List[str]
    keywords: List[str]


class PlannerAgent:
    """
    Planner Agent - coordinates multi-agent workflows.
    
    All inter-agent communication MUST go through the Planner.
    """
    
    def __init__(
        self,
        bedrock_client: BedrockClient = None,
        knowledge_base: KnowledgeBase = None,
    ):
        """
        Initialize Planner Agent.
        
        Args:
            bedrock_client: Bedrock client for decision-making
            knowledge_base: Knowledge base for context
        """
        self.bedrock_client = bedrock_client or BedrockClient()
        self.knowledge_base = knowledge_base
        self.agent_type = AgentType.PLANNER
        
        # Registry of available agents
        self.agent_capabilities = {
            AgentType.OBSERVABILITY: AgentCapability(
                agent_type=AgentType.OBSERVABILITY,
                capabilities=[
                    "metric interpretation",
                    "anomaly detection",
                    "bottleneck attribution",
                    "semantic insights",
                    "performance analysis"
                ],
                keywords=[
                    "metrics", "cpu", "memory", "latency", "throughput",
                    "anomaly", "spike", "performance", "slow"
                ]
            ),
            AgentType.INFRA: AgentCapability(
                agent_type=AgentType.INFRA,
                capabilities=[
                    "docker configuration analysis",
                    "ECS task analysis",
                    "infrastructure drift detection",
                    "configuration optimization"
                ],
                keywords=[
                    "docker", "dockerfile", "ecs", "container", "terraform",
                    "infrastructure", "configuration", "drift"
                ]
            ),
            AgentType.DB: AgentCapability(
                agent_type=AgentType.DB,
                capabilities=[
                    "query plan analysis",
                    "index recommendations",
                    "database optimization",
                    "schema analysis"
                ],
                keywords=[
                    "database", "postgres", "mysql", "redis", "query",
                    "index", "slow query", "sql", "schema"
                ]
            ),
            AgentType.COST: AgentCapability(
                agent_type=AgentType.COST,
                capabilities=[
                    "cost optimization",
                    "pricing analysis",
                    "right-sizing",
                    "cost projection"
                ],
                keywords=[
                    "cost", "price", "billing", "aws cost", "expensive",
                    "savings", "optimization", "budget", "inr"
                ]
            ),
            AgentType.CICD: AgentCapability(
                agent_type=AgentType.CICD,
                capabilities=[
                    "build time analysis",
                    "CI/CD regression detection",
                    "deployment tracking",
                    "build optimization"
                ],
                keywords=[
                    "cicd", "build", "github actions", "deployment",
                    "pipeline", "workflow", "commit", "build time"
                ]
            ),
        }
        
        # Agent instance registry (will be populated by dependency injection)
        self.agents = {}
        
        logger.info("Planner Agent initialized")
    
    def register_agent(self, agent_type: AgentType, agent_instance: Any):
        """
        Register a specialized agent.
        
        Args:
            agent_type: Type of agent
            agent_instance: Agent instance
        """
        self.agents[agent_type] = agent_instance
        logger.info(f"Registered agent: {agent_type.value}")
    
    def process_query(
        self,
        query: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process user query and coordinate agent execution.
        
        Args:
            query: User query or alert trigger
            context: Additional context
            
        Returns:
            Synthesized response from coordinated agents
        """
        start_time = time.time()
        task_id = str(uuid.uuid4())
        context = context or {}
        
        logger.info(f"Processing query (task_id={task_id}): {query}")
        
        try:
            # Determine which agents to invoke
            agent_plan = self._create_execution_plan(query, context)
            
            # Execute plan
            agent_responses = self._execute_plan(task_id, agent_plan, query, context)
            
            # Synthesize responses
            synthesized = self._synthesize_responses(query, agent_responses)
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'task_id': task_id,
                'query': query,
                'agents_invoked': [plan['agent'] for plan in agent_plan],
                'insights': synthesized['insights'],
                'recommendations': synthesized['recommendations'],
                'data': synthesized['data'],
                'execution_time_ms': execution_time,
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            logger.error(f"Query processing failed: {e}")
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'task_id': task_id,
                'query': query,
                'error': str(e),
                'execution_time_ms': execution_time,
                'status': 'FAILED'
            }
    
    def _create_execution_plan(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create execution plan - determine which agents to invoke and in what order.
        
        Args:
            query: User query
            context: Context
            
        Returns:
            Execution plan (list of agent invocations)
        """
        query_lower = query.lower()
        
        # Simple keyword-based routing (can be enhanced with LLM-based routing)
        plan = []
        
        for agent_type, capability in self.agent_capabilities.items():
            # Check if any keywords match
            if any(keyword in query_lower for keyword in capability.keywords):
                plan.append({
                    'agent': agent_type,
                    'priority': 1,  # Can be enhanced
                    'reason': f"Query matches {agent_type.value} keywords"
                })
        
        # If no agents matched, use LLM to determine routing
        if not plan:
            plan = self._llm_based_routing(query, context)
        
        # Sort by priority
        plan.sort(key=lambda x: x['priority'])
        
        logger.info(f"Execution plan: {[p['agent'].value for p in plan]}")
        return plan
    
    def _llm_based_routing(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Use LLM to determine which agents to invoke"""
        
        system_prompt = """You are a task router for a multi-agent AI SRE system.

Available agents:
- observability: Metric analysis, anomaly detection, performance insights
- infra: Docker/ECS configuration analysis, infrastructure drift
- db: Database query analysis, index recommendations
- cost: Cost optimization, pricing analysis
- cicd: Build time analysis, CI/CD regression detection

Determine which agent(s) should handle the query.

Return JSON:
{
  "agents": ["agent1", "agent2"],
  "reasoning": "why these agents"
}"""
        
        user_prompt = f"""Query: {query}

Context: {context}

Which agent(s) should handle this query?"""
        
        try:
            response = self.bedrock_client.invoke_with_json_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                use_haiku=True  # Fast routing
            )
            
            agents = response.get('agents', [])
            plan = []
            
            for agent_name in agents:
                try:
                    agent_type = AgentType(agent_name)
                    plan.append({
                        'agent': agent_type,
                        'priority': 1,
                        'reason': response.get('reasoning', 'LLM routing')
                    })
                except ValueError:
                    logger.warning(f"Unknown agent type: {agent_name}")
            
            return plan
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            # Fallback to observability agent
            return [{
                'agent': AgentType.OBSERVABILITY,
                'priority': 1,
                'reason': 'Fallback routing'
            }]
    
    def _execute_plan(
        self,
        task_id: str,
        plan: List[Dict[str, Any]],
        query: str,
        context: Dict[str, Any]
    ) -> List[AgentResponse]:
        """
        Execute the plan by invoking agents.
        
        Args:
            task_id: Task ID
            plan: Execution plan
            query: Original query
            context: Context
            
        Returns:
            List of agent responses
        """
        responses = []
        
        for step in plan:
            agent_type = step['agent']
            
            if agent_type not in self.agents:
                logger.warning(f"Agent {agent_type.value} not registered - skipping")
                continue
            
            agent = self.agents[agent_type]
            
            try:
                logger.info(f"Invoking {agent_type.value} agent...")
                
                # Create agent-specific task
                task_context = {
                    **context,
                    'query': query,
                    'planner_task_id': task_id,
                }
                
                # Different agents have different methods - call appropriate one
                # This is a simplified version - real implementation would have
                # standardized agent interface
                if hasattr(agent, 'analyze_metrics') and 'metrics' in context:
                    response = agent.analyze_metrics(
                        task_id=task_id,
                        metrics=context['metrics'],
                        context=task_context
                    )
                elif hasattr(agent, 'process_task'):
                    response = agent.process_task(task_id, query, task_context)
                else:
                    logger.warning(f"Agent {agent_type.value} has no standard method")
                    continue
                
                responses.append(response)
                
            except Exception as e:
                logger.error(f"Agent {agent_type.value} execution failed: {e}")
                # Create error response
                error_response = AgentResponse(
                    agent_type=agent_type,
                    task_id=task_id,
                    status=TaskStatus.FAILED,
                    insights=[],
                    data={},
                    execution_time_ms=0.0,
                    error_message=str(e)
                )
                responses.append(error_response)
        
        return responses
    
    def _synthesize_responses(
        self,
        query: str,
        responses: List[AgentResponse]
    ) -> Dict[str, Any]:
        """
        Synthesize multiple agent responses into coherent output.
        
        Args:
            query: Original query
            responses: Agent responses
            
        Returns:
            Synthesized response
        """
        all_insights = []
        all_data = {}
        
        for response in responses:
            if response.status != TaskStatus.FAILED:
                all_insights.extend(response.insights)
                all_data[response.agent_type.value] = response.data
        
        # Extract recommendations
        recommendations = []
        for insight in all_insights:
            recommendations.extend(insight.recommendations)
        
        # Deduplicate and prioritize
        unique_recommendations = list(set(recommendations))
        
        return {
            'insights': [
                {
                    'summary': insight.summary,
                    'business_impact': insight.business_impact,
                    'severity': insight.severity.value,
                    'confidence': insight.confidence_score,
                    'cost_impact_inr': insight.cost_impact_inr
                }
                for insight in all_insights
            ],
            'recommendations': unique_recommendations[:10],  # Top 10
            'data': all_data
        }
