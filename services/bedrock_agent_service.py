"""
AWS Bedrock Agents Service

Integrates with AWS SDK for Bedrock Agents to enable tool calling.
Provides structured tool definitions for agents to use.
"""

import logging
import json
import os
from typing import Dict, List, Optional, Any, Callable
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from config import config

logger = logging.getLogger(__name__)


class BedrockAgentTools:
    """
    AWS Bedrock Agent Tools integration.
    
    Provides tool definitions and execution for Bedrock Agents.
    """
    
    def __init__(
        self,
        region_name: str = None,
        agent_id: str = None,
        agent_alias_id: str = None,
    ):
        """
        Initialize Bedrock Agent Tools.
        
        Args:
            region_name: AWS region
            agent_id: Bedrock Agent ID (optional)
            agent_alias_id: Bedrock Agent Alias ID (optional)
        """
        self.region_name = region_name or config.AWS_REGION
        self.agent_id = agent_id or config.BEDROCK_AGENT_ID
        self.agent_alias_id = agent_alias_id or config.BEDROCK_AGENT_ALIAS_ID
        
        aws_config = Config(
            region_name=self.region_name,
            retries={'max_attempts': config.MAX_RETRIES, 'mode': 'adaptive'}
        )
        
        try:
            # Bedrock Agent Runtime for invoking agents with tools
            self.bedrock_agent_runtime = boto3.client(
                'bedrock-agent-runtime',
                config=aws_config
            )
            
            # CloudWatch for metrics
            self.cloudwatch = boto3.client('cloudwatch', config=aws_config)
            
            # ECS for infrastructure queries
            self.ecs = boto3.client('ecs', config=aws_config)
            
            # RDS for database operations
            self.rds = boto3.client('rds', config=aws_config)
            
            logger.info("Bedrock Agent Tools initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock Agent Tools: {e}")
            raise
    
    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for Bedrock Agents.
        
        Returns:
            List of tool definitions
        """
        return [
            {
                "toolSpec": {
                    "name": "get_cloudwatch_metrics",
                    "description": "Retrieve CloudWatch metrics for a given namespace, metric name, and dimensions",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "namespace": {
                                    "type": "string",
                                    "description": "CloudWatch namespace (e.g., AWS/ECS, AWS/RDS)"
                                },
                                "metric_name": {
                                    "type": "string",
                                    "description": "Metric name (e.g., CPUUtilization, MemoryUtilization)"
                                },
                                "dimensions": {
                                    "type": "object",
                                    "description": "Metric dimensions as key-value pairs"
                                },
                                "hours_back": {
                                    "type": "integer",
                                    "description": "How many hours of history to retrieve",
                                    "default": 24
                                }
                            },
                            "required": ["namespace", "metric_name"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "list_ecs_services",
                    "description": "List ECS services in a cluster",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "cluster_name": {
                                    "type": "string",
                                    "description": "ECS cluster name"
                                }
                            },
                            "required": ["cluster_name"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "describe_ecs_service",
                    "description": "Get detailed information about an ECS service",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "cluster_name": {
                                    "type": "string",
                                    "description": "ECS cluster name"
                                },
                                "service_name": {
                                    "type": "string",
                                    "description": "ECS service name"
                                }
                            },
                            "required": ["cluster_name", "service_name"]
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "list_rds_instances",
                    "description": "List RDS database instances",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "filters": {
                                    "type": "object",
                                    "description": "Optional filters for RDS instances"
                                }
                            }
                        }
                    }
                }
            },
            {
                "toolSpec": {
                    "name": "analyze_cost_anomaly",
                    "description": "Analyze AWS cost anomalies and trends",
                    "inputSchema": {
                        "json": {
                            "type": "object",
                            "properties": {
                                "service": {
                                    "type": "string",
                                    "description": "AWS service to analyze (optional)"
                                },
                                "days_back": {
                                    "type": "integer",
                                    "description": "How many days to analyze",
                                    "default": 30
                                }
                            }
                        }
                    }
                }
            },
        ]
    
    def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a tool with given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            parameters: Tool parameters
            
        Returns:
            Tool execution result
        """
        try:
            # Map tool names to execution methods
            tool_executors = {
                'get_cloudwatch_metrics': self._get_cloudwatch_metrics,
                'list_ecs_services': self._list_ecs_services,
                'describe_ecs_service': self._describe_ecs_service,
                'list_rds_instances': self._list_rds_instances,
                'analyze_cost_anomaly': self._analyze_cost_anomaly,
            }
            
            executor = tool_executors.get(tool_name)
            if not executor:
                return {
                    'success': False,
                    'error': f"Unknown tool: {tool_name}"
                }
            
            result = executor(parameters)
            return {
                'success': True,
                'data': result
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed for {tool_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _get_cloudwatch_metrics(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get CloudWatch metrics"""
        from datetime import datetime, timedelta
        
        namespace = params['namespace']
        metric_name = params['metric_name']
        dimensions = params.get('dimensions', {})
        hours_back = params.get('hours_back', 24)
        
        # Convert dimensions dict to CloudWatch format
        dimension_list = [
            {'Name': k, 'Value': v} 
            for k, v in dimensions.items()
        ]
        
        end_time = datetime.utcnow()
        start_time = end_time - timedelta(hours=hours_back)
        
        response = self.cloudwatch.get_metric_statistics(
            Namespace=namespace,
            MetricName=metric_name,
            Dimensions=dimension_list,
            StartTime=start_time,
            EndTime=end_time,
            Period=3600,  # 1 hour
            Statistics=['Average', 'Maximum', 'Minimum']
        )
        
        return {
            'metric_name': metric_name,
            'namespace': namespace,
            'datapoints': response.get('Datapoints', [])
        }
    
    def _list_ecs_services(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List ECS services"""
        cluster_name = params['cluster_name']
        
        response = self.ecs.list_services(cluster=cluster_name)
        
        return {
            'cluster': cluster_name,
            'service_arns': response.get('serviceArns', [])
        }
    
    def _describe_ecs_service(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Describe ECS service"""
        cluster_name = params['cluster_name']
        service_name = params['service_name']
        
        response = self.ecs.describe_services(
            cluster=cluster_name,
            services=[service_name]
        )
        
        services = response.get('services', [])
        if not services:
            return {'error': f"Service {service_name} not found"}
        
        service = services[0]
        
        return {
            'service_name': service.get('serviceName'),
            'status': service.get('status'),
            'desired_count': service.get('desiredCount'),
            'running_count': service.get('runningCount'),
            'pending_count': service.get('pendingCount'),
            'task_definition': service.get('taskDefinition'),
        }
    
    def _list_rds_instances(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """List RDS instances"""
        response = self.rds.describe_db_instances()
        
        instances = []
        for db_instance in response.get('DBInstances', []):
            instances.append({
                'identifier': db_instance.get('DBInstanceIdentifier'),
                'engine': db_instance.get('Engine'),
                'status': db_instance.get('DBInstanceStatus'),
                'instance_class': db_instance.get('DBInstanceClass'),
                'allocated_storage': db_instance.get('AllocatedStorage'),
            })
        
        return {
            'instances': instances,
            'count': len(instances)
        }
    
    def _analyze_cost_anomaly(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cost anomalies (placeholder)"""
        # This would integrate with AWS Cost Explorer
        # For now, return a placeholder
        return {
            'status': 'analysis_complete',
            'message': 'Cost anomaly analysis requires Cost Explorer API integration'
        }
    
    def invoke_agent_with_tools(
        self,
        session_id: str,
        input_text: str,
        enable_trace: bool = False
    ) -> Dict[str, Any]:
        """
        Invoke Bedrock Agent with tool calling capability.
        
        Args:
            session_id: Unique session identifier
            input_text: User input
            enable_trace: Enable trace for debugging
            
        Returns:
            Agent response with tool results
        """
        if not self.agent_id or not self.agent_alias_id:
            logger.warning("Bedrock Agent not configured")
            return {
                'success': False,
                'error': 'Bedrock Agent ID or Alias not configured'
            }
        
        try:
            response = self.bedrock_agent_runtime.invoke_agent(
                agentId=self.agent_id,
                agentAliasId=self.agent_alias_id,
                sessionId=session_id,
                inputText=input_text,
                enableTrace=enable_trace
            )
            
            # Process the event stream
            result_text = ""
            traces = []
            
            event_stream = response.get('completion', [])
            for event in event_stream:
                if 'chunk' in event:
                    chunk = event['chunk']
                    if 'bytes' in chunk:
                        result_text += chunk['bytes'].decode('utf-8')
                
                if 'trace' in event and enable_trace:
                    traces.append(event['trace'])
            
            return {
                'success': True,
                'response': result_text,
                'traces': traces if enable_trace else None,
                'session_id': session_id
            }
            
        except ClientError as e:
            logger.error(f"Failed to invoke Bedrock Agent: {e}")
            return {
                'success': False,
                'error': str(e)
            }


# Singleton instance
bedrock_agent_tools = BedrockAgentTools()
