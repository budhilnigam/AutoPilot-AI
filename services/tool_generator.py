"""
Tool Generator Service

Implements agentic recursion using Amazon Q Developer.
Dynamically generates tools that the system then uses.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import ast
import subprocess
import tempfile

logger = logging.getLogger(__name__)


class ToolGenerator:
    """
    Amazon Q-based tool generator for agentic recursion.
    
    Generates:
    - Infrastructure scanning tools
    - ECS deployment pipelines
    - IAM policy documents
    - SQL migration scripts
    - GitHub webhook listeners
    """
    
    def __init__(self, region_name: str = None):
        """
        Initialize Tool Generator.
        
        Args:
            region_name: AWS region
        """
        self.region_name = region_name or os.getenv('AWS_REGION', 'ap-south-1')
        
        # Note: Amazon Q Developer integration would go here
        # For now, we'll use Bedrock as a proxy for code generation
        config = Config(
            region_name=self.region_name,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        try:
            self.bedrock_runtime = boto3.client(
                'bedrock-runtime',
                config=config
            )
            logger.info("Tool Generator initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Tool Generator: {e}")
            raise
        
        # Directory for generated tools
        self.tools_dir = os.path.join(os.getcwd(), 'generated_tools')
        os.makedirs(self.tools_dir, exist_ok=True)
    
    def generate_infrastructure_scanner(
        self,
        resource_type: str,
        scan_parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an infrastructure scanning tool.
        
        Args:
            resource_type: AWS resource type (e.g., 'ec2', 'rds', 'lambda')
            scan_parameters: Parameters for scanning
            
        Returns:
            Generated tool metadata and code
        """
        prompt = f"""Generate a Python function to scan AWS {resource_type} resources.

Requirements:
- Use boto3
- Include error handling
- Return structured JSON
- Parameters: {json.dumps(scan_parameters)}
- Function name: scan_{resource_type}

Return only the Python code, no explanations."""
        
        code = self._generate_code(prompt)
        
        # Validate syntax
        if not self._validate_python_syntax(code):
            logger.error("Generated code has syntax errors")
            return {'status': 'FAILED', 'error': 'Syntax validation failed'}
        
        # Save tool
        filename = f"scan_{resource_type}.py"
        filepath = os.path.join(self.tools_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        logger.info(f"Generated infrastructure scanner: {filepath}")
        
        return {
            'status': 'SUCCESS',
            'tool_type': 'infrastructure_scanner',
            'resource_type': resource_type,
            'filepath': filepath,
            'code': code,
        }
    
    def generate_deployment_pipeline(
        self,
        service_name: str,
        deployment_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate ECS deployment pipeline script.
        
        Args:
            service_name: Service name
            deployment_config: Deployment configuration
            
        Returns:
            Generated pipeline metadata
        """
        prompt = f"""Generate a Python script for deploying {service_name} to AWS ECS.

Configuration:
{json.dumps(deployment_config, indent=2)}

Requirements:
- Use boto3 ECS client
- Include health checks
- Implement rollback on failure
- Add comprehensive error handling
- Function name: deploy_{service_name.replace('-', '_')}

Return only the Python code."""
        
        code = self._generate_code(prompt)
        
        if not self._validate_python_syntax(code):
            return {'status': 'FAILED', 'error': 'Syntax validation failed'}
        
        filename = f"deploy_{service_name}.py"
        filepath = os.path.join(self.tools_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(code)
        
        logger.info(f"Generated deployment pipeline: {filepath}")
        
        return {
            'status': 'SUCCESS',
            'tool_type': 'deployment_pipeline',
            'service_name': service_name,
            'filepath': filepath,
            'code': code,
        }
    
    def generate_iam_policy(
        self,
        policy_purpose: str,
        required_permissions: List[str]
    ) -> Dict[str, Any]:
        """
        Generate IAM policy document.
        
        Args:
            policy_purpose: Description of policy purpose
            required_permissions: List of required AWS actions
            
        Returns:
            Generated IAM policy
        """
        prompt = f"""Generate an AWS IAM policy document for: {policy_purpose}

Required permissions:
{json.dumps(required_permissions, indent=2)}

Requirements:
- Follow least privilege principle
- Include Resource ARN patterns
- Add conditions where appropriate
- Valid JSON format
- Include Version and Statement fields

Return only the JSON policy document."""
        
        policy = self._generate_json(prompt)
        
        # Validate policy structure
        if not self._validate_iam_policy(policy):
            return {'status': 'FAILED', 'error': 'Invalid IAM policy structure'}
        
        filename = f"policy_{policy_purpose.replace(' ', '_').lower()}.json"
        filepath = os.path.join(self.tools_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(policy, f, indent=2)
        
        logger.info(f"Generated IAM policy: {filepath}")
        
        return {
            'status': 'SUCCESS',
            'tool_type': 'iam_policy',
            'purpose': policy_purpose,
            'filepath': filepath,
            'policy': policy,
        }
    
    def generate_sql_migration(
        self,
        migration_description: str,
        schema_changes: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate SQL migration script.
        
        Args:
            migration_description: Description of migration
            schema_changes: Schema changes to apply
            
        Returns:
            Generated SQL script
        """
        prompt = f"""Generate a PostgreSQL migration script for: {migration_description}

Schema changes:
{json.dumps(schema_changes, indent=2)}

Requirements:
- Include transaction boundaries
- Add rollback statements
- Include index creation if needed
- Add comments
- Handle existing data migration if needed

Return only the SQL script."""
        
        sql = self._generate_code(prompt)
        
        filename = f"migration_{migration_description.replace(' ', '_').lower()}.sql"
        filepath = os.path.join(self.tools_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(sql)
        
        logger.info(f"Generated SQL migration: {filepath}")
        
        return {
            'status': 'SUCCESS',
            'tool_type': 'sql_migration',
            'description': migration_description,
            'filepath': filepath,
            'sql': sql,
        }
    
    def _generate_code(self, prompt: str) -> str:
        """Generate code using Bedrock (Claude)"""
        try:
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "temperature": 0.0,
                "system": "You are an expert code generator. Generate only clean, production-ready code with no explanations. Include proper error handling and type hints.",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            response = self.bedrock_runtime.invoke_model(
                modelId='anthropic.claude-3-5-sonnet-20241022-v2:0',
                body=json.dumps(request_body)
            )
            
            response_body = json.loads(response['body'].read())
            code = response_body['content'][0]['text']
            
            # Clean up code blocks
            code = code.strip()
            if code.startswith('```python'):
                code = code[9:]
            elif code.startswith('```'):
                code = code[3:]
            if code.endswith('```'):
                code = code[:-3]
            
            return code.strip()
            
        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            return ""
    
    def _generate_json(self, prompt: str) -> Dict[str, Any]:
        """Generate JSON using Bedrock"""
        code = self._generate_code(prompt)
        
        # Clean JSON from code blocks
        if code.startswith('```json'):
            code = code[7:]
        if code.endswith('```'):
            code = code[:-3]
        
        try:
            return json.loads(code.strip())
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse generated JSON: {e}")
            return {}
    
    def _validate_python_syntax(self, code: str) -> bool:
        """Validate Python code syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            logger.error(f"Python syntax error: {e}")
            return False
    
    def _validate_iam_policy(self, policy: Dict[str, Any]) -> bool:
        """Validate IAM policy structure"""
        required_fields = ['Version', 'Statement']
        
        if not all(field in policy for field in required_fields):
            logger.error("Missing required IAM policy fields")
            return False
        
        if not isinstance(policy['Statement'], list):
            logger.error("Statement must be a list")
            return False
        
        for statement in policy['Statement']:
            if 'Effect' not in statement or 'Action' not in statement:
                logger.error("Statement missing Effect or Action")
                return False
        
        return True
    
    def list_generated_tools(self) -> List[Dict[str, str]]:
        """List all generated tools"""
        tools = []
        
        for filename in os.listdir(self.tools_dir):
            filepath = os.path.join(self.tools_dir, filename)
            
            if os.path.isfile(filepath):
                ext = os.path.splitext(filename)[1]
                tools.append({
                    'filename': filename,
                    'filepath': filepath,
                    'type': ext[1:] if ext else 'unknown',
                })
        
        return tools
