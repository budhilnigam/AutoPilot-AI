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
        
        # Single-agent mode: route all tasks to the unified agent.
        self.agent_capabilities = {
            AgentType.UNIFIED: AgentCapability(
                agent_type=AgentType.UNIFIED,
                capabilities=[
                    "aws observability analysis",
                    "infrastructure guidance",
                    "database guidance",
                    "cost insights",
                    "ci/cd guidance",
                    "cross-domain synthesis",
                ],
                keywords=[],
            )
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
            primary_agent = agent_plan[0]['agent'].value if agent_plan else self.agent_type.value
            
            execution_time = (time.time() - start_time) * 1000
            
            return {
                'task_id': task_id,
                'query': query,
                'agents_invoked': [plan['agent'] for plan in agent_plan],
                'summary': synthesized.get('summary', ''),
                'insights': synthesized['insights'],
                'recommendations': synthesized['recommendations'],
                'data': synthesized['data'],
                'agent_type': primary_agent,
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
        # Single-agent execution plan.
        if AgentType.UNIFIED in self.agents:
            plan = [{
                'agent': AgentType.UNIFIED,
                'priority': 1,
                'reason': 'Single-agent mode'
            }]
        elif AgentType.OBSERVABILITY in self.agents:
            # Backward compatibility fallback if unified agent is not registered.
            plan = [{
                'agent': AgentType.OBSERVABILITY,
                'priority': 1,
                'reason': 'Single-agent fallback mode'
            }]
        else:
            plan = self._llm_based_routing(query, context)

        logger.info(f"Execution plan: {[p['agent'].value for p in plan]}")
        return plan
    
    def _llm_based_routing(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Use LLM to determine which agents to invoke"""

        system_prompt = """You are a task router for a single-agent AI SRE system.

Always route the query to the unified agent.

Return JSON:
{
    "agents": ["unified"],
    "reasoning": "single-agent mode"
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
            
            if not plan:
                return [{
                    'agent': AgentType.UNIFIED,
                    'priority': 1,
                    'reason': 'Single-agent fallback'
                }]

            return plan
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            # Fallback to unified agent
            return [{
                'agent': AgentType.UNIFIED,
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
                elif agent_type in (AgentType.OBSERVABILITY, AgentType.UNIFIED) and hasattr(agent, 'analyze_with_dynamic_tools'):
                    response = agent.analyze_with_dynamic_tools(
                        task_id=task_id,
                        user_query=query,
                        context=task_context,
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

        # Build summary from agent response data, not from insight summaries  
        summary = None
        
        # Try to get actual response text from agent data
        for response in responses:
            if response.status != TaskStatus.FAILED and response.data:
                # Check for full_response in data
                if isinstance(response.data, dict):
                    full_response = response.data.get('full_response')
                    if full_response:
                        summary = full_response
                        break
                    # Try answer field
                    answer = response.data.get('answer')
                    if answer:
                        summary = answer
                        break
        
        # Fallback: create a brief intro summary without duplicating insights
        if not summary:
            if all_insights:
                # Get the primary agent type
                primary_agent_type = responses[0].agent_type.value if responses else 'system'
                insight_count = len(all_insights)
                
                # Create a brief intro that doesn't duplicate insight content
                if insight_count == 1:
                    summary = f"Completed {primary_agent_type} analysis. See detailed insight below."
                else:
                    summary = f"Completed {primary_agent_type} analysis. Found {insight_count} items (see details below)."
            elif responses:
                failed_agents = [
                    response.agent_type.value
                    for response in responses
                    if response.status == TaskStatus.FAILED
                ]
                if failed_agents:
                    summary = (
                        "I could not complete this request because agent execution failed for: "
                        + ", ".join(failed_agents)
                        + "."
                    )
                else:
                    summary = "I could not find actionable results for this request."
            else:
                summary = "No agent was selected to handle this request."
        
        return {
            'summary': summary,
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
