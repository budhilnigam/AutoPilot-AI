"""
Amazon Bedrock Client

Wrapper for Amazon Bedrock Runtime API.
Handles Claude model invocations with structured output parsing.
"""

import json
import logging
import os
from typing import Dict, List, Optional, Any
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)


class BedrockClient:
    """
    Amazon Bedrock client for Claude model invocations.
    
    Provides structured interface for:
    - Claude Sonnet (deep reasoning)
    - Claude Haiku (fast classification)
    """
    
    def __init__(
        self,
        region_name: str = None,
        model_id: str = None,
        haiku_model_id: str = None,
    ):
        """
        Initialize Bedrock client.
        
        Args:
            region_name: AWS region (defaults to env AWS_REGION or ap-south-1)
            model_id: Bedrock model ID for Sonnet (defaults to env BEDROCK_MODEL_ID)
            haiku_model_id: Bedrock model ID for Haiku (defaults to env BEDROCK_HAIKU_MODEL_ID)
        """
        self.region_name = region_name or os.getenv('AWS_REGION', 'ap-south-1')
        self.model_id = model_id or os.getenv(
            'BEDROCK_MODEL_ID',
            'anthropic.claude-3-5-sonnet-20241022-v2:0'
        )
        self.haiku_model_id = haiku_model_id or os.getenv(
            'BEDROCK_HAIKU_MODEL_ID',
            'anthropic.claude-3-haiku-20240307-v1:0'
        )
        
        # Configure boto3 client with retries
        config = Config(
            region_name=self.region_name,
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        try:
            self.bedrock_runtime = boto3.client(
                service_name='bedrock-runtime',
                config=config
            )
            logger.info(f"Bedrock client initialized for region {self.region_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Bedrock client: {e}")
            raise
    
    def invoke_claude_sonnet(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        stop_sequences: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Invoke Claude Sonnet for deep reasoning tasks.
        
        Args:
            system_prompt: System context and instructions
            user_prompt: User query/task
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            stop_sequences: Optional stop sequences
            
        Returns:
            Dict with 'content' and 'stop_reason'
        """
        return self._invoke_claude(
            model_id=self.model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )
    
    def invoke_claude_haiku(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        stop_sequences: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Invoke Claude Haiku for fast classification tasks.
        
        Args:
            system_prompt: System context and instructions
            user_prompt: User query/task
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            stop_sequences: Optional stop sequences
            
        Returns:
            Dict with 'content' and 'stop_reason'
        """
        return self._invoke_claude(
            model_id=self.haiku_model_id,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop_sequences=stop_sequences,
        )
    
    def _invoke_claude(
        self,
        model_id: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        max_tokens: int,
        stop_sequences: Optional[List[str]],
    ) -> Dict[str, Any]:
        """
        Internal method to invoke Claude models.
        
        Args:
            model_id: Bedrock model identifier
            system_prompt: System context
            user_prompt: User query
            temperature: Sampling temperature
            max_tokens: Max response tokens
            stop_sequences: Stop sequences
            
        Returns:
            Response dict with content and metadata
            
        Raises:
            ClientError: If Bedrock API call fails
            ValueError: If response parsing fails
        """
        try:
            # Construct request body for Claude 3+ models
            request_body = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": max_tokens,
                "temperature": temperature,
                "system": system_prompt,
                "messages": [
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ]
            }
            
            if stop_sequences:
                request_body["stop_sequences"] = stop_sequences
            
            logger.debug(f"Invoking {model_id} with {len(user_prompt)} char prompt")
            
            # Invoke model
            response = self.bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract content from Claude response
            content = ""
            if 'content' in response_body and len(response_body['content']) > 0:
                content = response_body['content'][0].get('text', '')
            
            result = {
                'content': content,
                'stop_reason': response_body.get('stop_reason', 'unknown'),
                'usage': response_body.get('usage', {}),
            }
            
            logger.info(
                f"Successfully invoked {model_id}. "
                f"Input tokens: {result['usage'].get('input_tokens', 0)}, "
                f"Output tokens: {result['usage'].get('output_tokens', 0)}"
            )
            
            return result
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            logger.error(f"Bedrock API error ({error_code}): {error_message}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error invoking Bedrock: {e}")
            raise
    
    def parse_json_response(self, response_content: str) -> Dict[str, Any]:
        """
        Parse JSON from Claude response.
        
        Handles common formatting issues:
        - Markdown code blocks
        - Leading/trailing whitespace
        - Partial JSON
        
        Args:
            response_content: Raw response from Claude
            
        Returns:
            Parsed JSON as dict
            
        Raises:
            ValueError: If JSON parsing fails
        """
        try:
            # Remove markdown code blocks if present
            content = response_content.strip()
            if content.startswith('```json'):
                content = content[7:]  # Remove ```json
            elif content.startswith('```'):
                content = content[3:]  # Remove ```
            if content.endswith('```'):
                content = content[:-3]  # Remove closing ```
            
            content = content.strip()
            
            # Parse JSON
            parsed = json.loads(content)
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw content: {response_content[:500]}")
            raise ValueError(f"Invalid JSON in Claude response: {e}")
    
    def invoke_with_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        use_haiku: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke Claude and parse JSON response.
        
        Convenience method that combines invocation and JSON parsing.
        
        Args:
            system_prompt: System prompt
            user_prompt: User prompt
            use_haiku: Use Haiku model (faster) instead of Sonnet
            **kwargs: Additional arguments for invoke_claude_*
            
        Returns:
            Parsed JSON response
        """
        if use_haiku:
            response = self.invoke_claude_haiku(system_prompt, user_prompt, **kwargs)
        else:
            response = self.invoke_claude_sonnet(system_prompt, user_prompt, **kwargs)
        
        return self.parse_json_response(response['content'])
