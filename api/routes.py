"""
API Routes for AutoPilot AI

Provides REST API endpoints for interacting with the multi-agent system.
"""

import logging
from typing import Dict, Any, List
from datetime import datetime

from models.core_models import MetricData, MetricType, Configuration, BuildData, QueryPattern
from models.agent_protocol import AgentType
from agents.planner_agent import PlannerAgent
from agents.observability_agent import ObservabilityAgent
from agents.infra_agent import InfraAgent
from agents.db_agent import DBAgent
from agents.cost_agent import CostAgent
from agents.cicd_agent import CICDAgent
from services.bedrock_client import BedrockClient
from services.knowledge_base_factory import KnowledgeBaseInterface
from services.cloudwatch_client import CloudWatchClient
from services.billing_client import BillingClient
from services.github_client import GitHubClient
from services.tool_generator import ToolGenerator

logger = logging.getLogger(__name__)


class AutoPilotAPI:
    """
    AutoPilot AI API
    
    Provides endpoints for querying the multi-agent system.
    """
    
    def __init__(self):
        """Initialize API with all agents"""
        logger.info("Initializing AutoPilot AI API...")
        
        # Initialize services
        self.bedrock_client = BedrockClient()
        self.knowledge_base = None
        try:
            self.knowledge_base = KnowledgeBaseInterface()
        except Exception as e:
            logger.warning(f"Knowledge Base initialization failed: {e}")
        
        self.cloudwatch_client = CloudWatchClient()
        self.billing_client = BillingClient()
        self.github_client = GitHubClient()
        self.tool_generator = ToolGenerator()
        
        # Initialize agents
        self.observability_agent = ObservabilityAgent(
            bedrock_client=self.bedrock_client,
            knowledge_base=self.knowledge_base,
            cloudwatch_client=self.cloudwatch_client
        )
        
        self.infra_agent = InfraAgent(
            bedrock_client=self.bedrock_client,
            knowledge_base=self.knowledge_base
        )
        
        self.db_agent = DBAgent(
            bedrock_client=self.bedrock_client,
            knowledge_base=self.knowledge_base
        )
        
        self.cost_agent = CostAgent(
            bedrock_client=self.bedrock_client,
            knowledge_base=self.knowledge_base,
            billing_client=self.billing_client
        )
        
        self.cicd_agent = CICDAgent(
            bedrock_client=self.bedrock_client,
            knowledge_base=self.knowledge_base,
            github_client=self.github_client
        )
        
        # Initialize Planner Agent and register specialized agents
        self.planner_agent = PlannerAgent(
            bedrock_client=self.bedrock_client,
            knowledge_base=self.knowledge_base
        )
        
        self.planner_agent.register_agent(AgentType.OBSERVABILITY, self.observability_agent)
        self.planner_agent.register_agent(AgentType.INFRA, self.infra_agent)
        self.planner_agent.register_agent(AgentType.DB, self.db_agent)
        self.planner_agent.register_agent(AgentType.COST, self.cost_agent)
        self.planner_agent.register_agent(AgentType.CICD, self.cicd_agent)
        
        logger.info("AutoPilot AI API initialized successfully")
    
    def query(self, user_query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a natural language query.
        
        Args:
            user_query: Natural language query
            context: Additional context
            
        Returns:
            Response with insights and recommendations
        """
        logger.info(f"Processing query: {user_query}")
        return self.planner_agent.process_query(user_query, context)
    
    def analyze_metrics(
        self,
        metrics: List[Dict[str, Any]],
        description: str = "Metric analysis"
    ) -> Dict[str, Any]:
        """
        Analyze metrics and generate insights.
        
        Args:
            metrics: List of metric dictionaries
            description: Analysis description
            
        Returns:
            Insights and recommendations
        """
        # Convert dict metrics to MetricData objects
        metric_objects = []
        for m in metrics:
            metric = MetricData(
                metric_name=m['metric_name'],
                metric_type=MetricType(m.get('metric_type', 'custom')),
                value=float(m['value']),
                unit=m.get('unit', 'None'),
                timestamp=m.get('timestamp', datetime.utcnow().isoformat()),
                dimensions=m.get('dimensions', {}),
                source=m.get('source', 'api')
            )
            metric_objects.append(metric)
        
        context = {
            'metrics': metric_objects,
            'description': description
        }
        
        return self.planner_agent.process_query(
            f"Analyze these metrics: {description}",
            context
        )
    
    def analyze_configuration(
        self,
        config_type: str,
        config_content: str,
        metadata: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Analyze infrastructure configuration.
        
        Args:
            config_type: Type of configuration (dockerfile, ecs_task, etc.)
            config_content: Configuration content
            metadata: Additional metadata
            
        Returns:
            Configuration analysis insights
        """
        import hashlib
        
        config = Configuration(
            config_type=config_type,
            config_content=config_content,
            config_hash=hashlib.md5(config_content.encode()).hexdigest(),
            timestamp=datetime.utcnow().isoformat(),
            source='api',
            metadata=metadata or {}
        )
        
        context = {
            'configuration': config
        }
        
        return self.planner_agent.process_query(
            f"Analyze {config_type} configuration",
            context
        )
    
    def analyze_database_query(
        self,
        query_template: str,
        execution_count: int,
        avg_duration_ms: float,
        tables_accessed: List[str]
    ) -> Dict[str, Any]:
        """
        Analyze database query performance.
        
        Args:
            query_template: SQL query template
            execution_count: Number of executions
            avg_duration_ms: Average duration in milliseconds
            tables_accessed: Tables accessed by query
            
        Returns:
            Query optimization insights
        """
        query_pattern = QueryPattern(
            query_template=query_template,
            execution_count=execution_count,
            avg_duration_ms=avg_duration_ms,
            max_duration_ms=avg_duration_ms * 1.5,  # Estimate
            tables_accessed=tables_accessed
        )
        
        context = {
            'query_pattern': query_pattern
        }
        
        return self.planner_agent.process_query(
            "Analyze database query performance",
            context
        )
    
    def analyze_build(
        self,
        repository: str,
        build_id: str = None
    ) -> Dict[str, Any]:
        """
        Analyze CI/CD build performance.
        
        Args:
            repository: Repository (owner/repo)
            build_id: Optional specific build ID
            
        Returns:
            Build analysis insights
        """
        context = {
            'repository': repository
        }
        
        if build_id:
            context['build_id'] = build_id
        
        return self.planner_agent.process_query(
            f"Analyze CI/CD builds for {repository}",
            context
        )
    
    def get_cost_analysis(self) -> Dict[str, Any]:
        """
        Get cost analysis and optimization recommendations.
        
        Returns:
            Cost insights in INR
        """
        return self.planner_agent.process_query(
            "Analyze AWS costs and provide optimization recommendations",
            {}
        )
    
    def generate_tool(
        self,
        tool_type: str,
        tool_spec: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a tool using Amazon Q / Bedrock.
        
        Args:
            tool_type: Type of tool (infrastructure_scanner, deployment_pipeline, etc.)
            tool_spec: Tool specification
            
        Returns:
            Generated tool metadata
        """
        if tool_type == 'infrastructure_scanner':
            return self.tool_generator.generate_infrastructure_scanner(
                resource_type=tool_spec['resource_type'],
                scan_parameters=tool_spec.get('parameters', {})
            )
        elif tool_type == 'deployment_pipeline':
            return self.tool_generator.generate_deployment_pipeline(
                service_name=tool_spec['service_name'],
                deployment_config=tool_spec.get('config', {})
            )
        elif tool_type == 'iam_policy':
            return self.tool_generator.generate_iam_policy(
                policy_purpose=tool_spec['purpose'],
                required_permissions=tool_spec.get('permissions', [])
            )
        elif tool_type == 'sql_migration':
            return self.tool_generator.generate_sql_migration(
                migration_description=tool_spec['description'],
                schema_changes=tool_spec.get('schema_changes', {})
            )
        else:
            return {
                'status': 'FAILED',
                'error': f'Unknown tool type: {tool_type}'
            }
    
    def list_generated_tools(self) -> List[Dict[str, str]]:
        """List all generated tools"""
        return self.tool_generator.list_generated_tools()
