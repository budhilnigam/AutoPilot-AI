"""
Dynamic AWS API Executor

Enables LLM agents to execute arbitrary AWS SDK operations dynamically.
This eliminates the need to hardcode every possible AWS API call.
"""

import logging
import json
from typing import Dict, Any, Optional, List
import boto3
from botocore.exceptions import ClientError, ParamValidationError
from botocore.config import Config

logger = logging.getLogger(__name__)


class AWSAPIExecutor:
    """
    Generic AWS API executor for dynamic agent tool usage.
    
    Allows LLM agents to make any AWS SDK call by specifying:
    - service name (e.g., 'ec2', 's3', 'cloudwatch')
    - operation name (e.g., 'describe_instances', 'list_buckets', 'list_metrics')
    - parameters (dict)
    
    The LLM decides what to call based on the user query.
    """
    
    def __init__(self, region_name: str = 'us-east-1', max_retries: int = 3):
        """
        Initialize AWS API Executor.
        
        Args:
            region_name: AWS region for API calls
            max_retries: Maximum retry attempts for failed calls
        """
        self.region_name = region_name
        self.config = Config(
            region_name=region_name,
            retries={'max_attempts': max_retries, 'mode': 'adaptive'}
        )
        self._client_cache: Dict[str, Any] = {}
        logger.info(f"AWS API Executor initialized for region {region_name}")
    
    def _get_client(self, service_name: str) -> Any:
        """
        Get or create a boto3 client for the specified service.
        Uses caching to avoid recreating clients.
        
        Args:
            service_name: AWS service name (e.g., 'ec2', 's3', 'cloudwatch')
            
        Returns:
            Boto3 client for the service
        """
        if service_name not in self._client_cache:
            try:
                self._client_cache[service_name] = boto3.client(
                    service_name,
                    config=self.config
                )
                logger.debug(f"Created boto3 client for service: {service_name}")
            except Exception as e:
                logger.error(f"Failed to create client for {service_name}: {e}")
                raise
        
        return self._client_cache[service_name]
    
    def execute(
        self,
        service: str,
        operation: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute an AWS API operation dynamically.
        
        Args:
            service: AWS service name (e.g., 'ec2', 's3', 'cloudwatch', 'ce')
            operation: API operation name (e.g., 'describe_instances', 'list_metrics')
            parameters: Dictionary of parameters for the operation (optional)
            
        Returns:
            API response as dictionary
            
        Raises:
            ClientError: If AWS API returns an error
            ParamValidationError: If parameters are invalid
            AttributeError: If operation doesn't exist on the service
            
        Example:
            # List CloudWatch metrics
            result = executor.execute(
                service='cloudwatch',
                operation='list_metrics',
                parameters={'Namespace': 'AWS/EC2', 'MetricName': 'CPUUtilization'}
            )
            
            # Get EC2 instances
            result = executor.execute(
                service='ec2',
                operation='describe_instances',
                parameters={}
            )
            
            # Get monthly costs
            result = executor.execute(
                service='ce',
                operation='get_cost_and_usage',
                parameters={
                    'TimePeriod': {'Start': '2026-03-01', 'End': '2026-03-08'},
                    'Granularity': 'MONTHLY',
                    'Metrics': ['UnblendedCost']
                }
            )
        """
        if parameters is None:
            parameters = {}
        
        try:
            # Get the client for this service
            client = self._get_client(service)
            
            # Get the operation method dynamically
            if not hasattr(client, operation):
                available_ops = [
                    method for method in dir(client)
                    if not method.startswith('_') and callable(getattr(client, method))
                ]
                logger.error(
                    f"Operation '{operation}' not found on service '{service}'. "
                    f"Available operations: {', '.join(available_ops[:10])}..."
                )
                raise AttributeError(
                    f"Service '{service}' does not have operation '{operation}'"
                )
            
            operation_method = getattr(client, operation)
            
            # Execute the operation
            logger.info(
                f"Executing AWS API: {service}.{operation} with params: "
                f"{json.dumps(parameters, default=str)[:200]}"
            )
            
            response = operation_method(**parameters)
            
            # Remove ResponseMetadata for cleaner responses (optional)
            if isinstance(response, dict) and 'ResponseMetadata' in response:
                response_copy = response.copy()
                del response_copy['ResponseMetadata']
                logger.debug(f"API call successful: {service}.{operation}")
                return response_copy
            
            return response
            
        except ParamValidationError as e:
            logger.error(f"Parameter validation failed for {service}.{operation}: {e}")
            raise ValueError(
                f"Invalid parameters for {service}.{operation}: {str(e)}"
            )
        
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', 'Unknown')
            error_message = e.response.get('Error', {}).get('Message', str(e))
            logger.error(
                f"AWS API error for {service}.{operation}: [{error_code}] {error_message}"
            )
            raise
        
        except Exception as e:
            logger.error(f"Unexpected error executing {service}.{operation}: {e}")
            raise
    
    def get_tool_definition(self) -> Dict[str, Any]:
        """
        Get the tool definition for LLM function calling.
        This allows the LLM to use this executor as a tool.
        
        Returns:
            Tool definition in Anthropic tool format
        """
        return {
            "name": "aws_api_executor",
            "description": (
                "Execute AWS SDK operations dynamically. Use this to interact with any AWS service. "
                "You can call any boto3 client operation. Common services: ec2, s3, cloudwatch, ce (Cost Explorer), "
                "rds, lambda, dynamodb, ecs, logs, sts. "
                "IMPORTANT: Do not include 'MaxRecords' parameter for CloudWatch list_metrics - it's not supported. "
                "Use pagination with 'NextToken' if needed."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": (
                            "AWS service name in lowercase (e.g., 'ec2', 's3', 'cloudwatch', 'ce', 'rds', 'lambda'). "
                            "This is the boto3 client name."
                        )
                    },
                    "operation": {
                        "type": "string",
                        "description": (
                            "AWS API operation name in snake_case (e.g., 'describe_instances', 'list_metrics', "
                            "'get_cost_and_usage'). This is the boto3 method name."
                        )
                    },
                    "parameters": {
                        "type": "object",
                        "description": (
                            "Parameters for the API operation as a dictionary. Use exact parameter names from AWS SDK. "
                            "Example for list_metrics: {'Namespace': 'AWS/EC2', 'MetricName': 'CPUUtilization'}. "
                            "Leave empty {} if no parameters needed."
                        )
                    }
                },
                "required": ["service", "operation"]
            }
        }
    
    def list_available_services(self) -> List[str]:
        """
        List commonly used AWS services available for execution.
        
        Returns:
            List of service names
        """
        return [
            'ec2',           # EC2 instances
            's3',            # S3 storage
            'cloudwatch',    # Metrics and monitoring
            'logs',          # CloudWatch Logs
            'ce',            # Cost Explorer
            'rds',           # Relational Database Service
            'dynamodb',      # DynamoDB
            'lambda',        # Lambda functions
            'ecs',           # Elastic Container Service
            'iam',           # Identity and Access Management
            'sns',           # Simple Notification Service
            'sqs',           # Simple Queue Service
            'elasticache',   # ElastiCache (Redis/Memcached)
            'route53',       # DNS service
            'elb',           # Elastic Load Balancing (Classic)
            'elbv2',         # Elastic Load Balancing v2 (ALB/NLB)
            'autoscaling',   # Auto Scaling
            'cloudformation',# CloudFormation stacks
            'sts',           # Security Token Service
            'organizations', # AWS Organizations
        ]


# Singleton instance for global use
_executor_instance: Optional[AWSAPIExecutor] = None


def get_aws_executor(region_name: str = 'us-east-1') -> AWSAPIExecutor:
    """
    Get the global AWS API executor instance (singleton pattern).
    
    Args:
        region_name: AWS region (only used when creating new instance)
        
    Returns:
        AWSAPIExecutor instance
    """
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = AWSAPIExecutor(region_name=region_name)
    return _executor_instance
