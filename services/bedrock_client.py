"""
Amazon Bedrock Client

Wrapper for Amazon Bedrock Runtime API.
Handles Claude model invocations with structured output parsing.
"""

import json
import logging
import os
import re
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
            region_name: AWS region (defaults to env AWS_REGION or us-east-1)
            model_id: Bedrock model ID for Sonnet (defaults to env BEDROCK_MODEL_ID)
            haiku_model_id: Bedrock model ID for Haiku (defaults to env BEDROCK_HAIKU_MODEL_ID)
        """
        self.region_name = region_name or os.getenv('AWS_REGION', 'us-east-1')
        self.model_id = model_id or os.getenv(
            'BEDROCK_MODEL_ID',
            'openai.gpt-oss-20b-1:0'
        )
        self.haiku_model_id = haiku_model_id or os.getenv(
            'BEDROCK_HAIKU_MODEL_ID',
            'openai.gpt-oss-20b-1:0'
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

    def _is_openai_model(self, model_id: str) -> bool:
        """Return True when using a Bedrock OpenAI-compatible model."""
        return model_id.startswith('openai.')

    def _extract_text_from_response_body(self, response_body: Dict[str, Any]) -> str:
        """Extract text across Anthropic-style and OpenAI-style Bedrock responses."""
        # Anthropic/Claude style: content=[{"type":"text","text":"..."}]
        if isinstance(response_body.get('content'), list):
            text_parts = []
            for block in response_body.get('content', []):
                if isinstance(block, dict):
                    if block.get('type') == 'text':
                        text_parts.append(block.get('text', ''))
                    elif isinstance(block.get('text'), str):
                        text_parts.append(block.get('text', ''))
            if text_parts:
                return ''.join(text_parts)

        # OpenAI style: choices[0].message.content
        choices = response_body.get('choices', [])
        if choices and isinstance(choices[0], dict):
            message = choices[0].get('message', {})
            content = message.get('content', '')
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get('type') == 'text':
                        parts.append(item.get('text', ''))
                return ''.join(parts)

        # Last resort
        if isinstance(response_body.get('output_text'), str):
            return response_body['output_text']

        return ''
    
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
            
            # Extract content from response (supports multiple Bedrock model schemas)
            content = self._extract_text_from_response_body(response_body)

            usage = response_body.get('usage', {})
            if not usage:
                usage = response_body.get('token_usage', {})
            
            result = {
                'content': content,
                'stop_reason': response_body.get('stop_reason') or response_body.get('finish_reason', 'unknown'),
                'usage': usage,
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
        Parse JSON from LLM response.
        
        Handles common formatting issues:
        - Markdown code blocks
        - Leading/trailing whitespace
        - Partial JSON
        
        Args:
            response_content: Raw response from LLM
            
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
            
            # Parse JSON directly first
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                pass

            # Recover JSON object/array embedded in surrounding text
            object_match = re.search(r'\{[\s\S]*\}', content)
            array_match = re.search(r'\[[\s\S]*\]', content)
            candidate = None
            if object_match and array_match:
                candidate = object_match.group(0) if object_match.start() < array_match.start() else array_match.group(0)
            elif object_match:
                candidate = object_match.group(0)
            elif array_match:
                candidate = array_match.group(0)

            if candidate:
                return json.loads(candidate)

            raise json.JSONDecodeError('No JSON object or array found', content, 0)
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Raw content: {response_content[:500]}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")
    
    def invoke_with_json_response(
        self,
        system_prompt: str,
        user_prompt: str,
        use_haiku: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke LLM and parse JSON response.
        
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
    
    def invoke_with_tools(
        self,
        system_prompt: str,
        user_prompt: str,
        tools: List[Dict[str, Any]],
        tool_executor: callable,
        max_iterations: int = 5,
        max_output_tokens: int = 1200,
        temperature: float = 0.0,
        use_haiku: bool = False,
    ) -> Dict[str, Any]:
        """
        Invoke Claude with tool support (function calling).
        
        This enables the LLM to:
        1. Decide which tools to use based on the user query
        2. Call tools with appropriate parameters
        3. Use tool results to formulate a final answer
        
        Args:
            system_prompt: System context and instructions
            user_prompt: User query/task
            tools: List of tool definitions (Anthropic tool format)
            tool_executor: Callable that executes tools. Signature: (tool_name, tool_input) -> result
            max_iterations: Maximum tool use iterations to prevent infinite loops
            max_output_tokens: Maximum tokens in each model response for tool loop iterations
            temperature: Sampling temperature
            use_haiku: Use Haiku model (faster) instead of Sonnet
            
        Returns:
            Dict with final response and tool usage metadata
            
        Example:
            def execute_tool(tool_name, tool_input):
                if tool_name == "aws_api_executor":
                    executor = AWSAPIExecutor()
                    return executor.execute(**tool_input)
                    
            result = client.invoke_with_tools(
                system_prompt="You are an AWS expert assistant.",
                user_prompt="List all EC2 instances",
                tools=[aws_executor.get_tool_definition()],
                tool_executor=execute_tool
            )
        """
        model_id = self.haiku_model_id if use_haiku else self.model_id
        messages = [{"role": "user", "content": user_prompt}]
        iteration = 0
        
        logger.info(f"Starting tool-enabled invocation with {len(tools)} tools available")
        
        while iteration < max_iterations:
            iteration += 1
            
            try:
                # Construct request with tools (model-specific schema)
                if self._is_openai_model(model_id):
                    openai_tools = []
                    for tool in tools:
                        openai_tools.append({
                            "type": "function",
                            "function": {
                                "name": tool.get("name"),
                                "description": tool.get("description", ""),
                                "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                            },
                        })

                    request_body = {
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            *messages,
                        ],
                        "max_completion_tokens": max_output_tokens,
                        "temperature": temperature,
                        "tools": openai_tools,
                        "tool_choice": "auto",
                    }
                else:
                    request_body = {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": max_output_tokens,
                        "temperature": temperature,
                        "system": system_prompt,
                        "messages": messages,
                        "tools": tools,
                    }
                
                logger.debug(f"Tool invocation iteration {iteration}/{max_iterations}")
                
                # Invoke model
                response = self.bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(request_body)
                )
                
                response_body = json.loads(response['body'].read())
                stop_reason = response_body.get('stop_reason')

                if self._is_openai_model(model_id):
                    choices = response_body.get('choices', [])
                    choice = choices[0] if choices else {}
                    stop_reason = choice.get('finish_reason', stop_reason)
                    message = choice.get('message', {})
                    tool_calls = message.get('tool_calls', []) if isinstance(message, dict) else []

                    if tool_calls:
                        tool_results = []
                        for call in tool_calls:
                            function_payload = call.get('function', {})
                            tool_name = function_payload.get('name')
                            raw_args = function_payload.get('arguments', '{}')
                            try:
                                tool_input = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                            except json.JSONDecodeError:
                                tool_input = {}

                            logger.info(
                                f"Executing tool: {tool_name} with input: {json.dumps(tool_input, default=str)[:200]}"
                            )

                            try:
                                tool_result = tool_executor(tool_name, tool_input)
                                tool_results.append({
                                    "role": "tool",
                                    "tool_call_id": call.get('id'),
                                    "content": json.dumps(tool_result, default=str),
                                })
                            except Exception as tool_error:
                                logger.error(f"Tool execution error: {tool_error}")
                                tool_results.append({
                                    "role": "tool",
                                    "tool_call_id": call.get('id'),
                                    "content": f"Error executing tool: {str(tool_error)}",
                                })

                        messages.append({
                            "role": "assistant",
                            "content": message.get('content', '') if isinstance(message, dict) else '',
                            "tool_calls": tool_calls,
                        })
                        messages.extend(tool_results)
                        continue

                    final_text = self._extract_text_from_response_body(response_body)
                    return {
                        'content': final_text or 'No response generated',
                        'stop_reason': stop_reason or 'end_turn',
                        'usage': response_body.get('usage', {}),
                        'iterations': iteration,
                        'messages': messages,
                    }
                
                # Add assistant response to message history
                assistant_message = {
                    "role": "assistant",
                    "content": response_body.get('content', [])
                }
                messages.append(assistant_message)
                
                logger.debug(f"Stop reason: {stop_reason}")
                
                # Check if tool use is requested
                if stop_reason == 'tool_use':
                    # Extract tool use requests
                    tool_results = []
                    
                    for content_block in response_body.get('content', []):
                        if content_block.get('type') == 'tool_use':
                            tool_name = content_block['name']
                            tool_input = content_block['input']
                            tool_use_id = content_block['id']
                            
                            logger.info(f"Executing tool: {tool_name} with input: {json.dumps(tool_input, default=str)[:200]}")
                            
                            # Execute the tool
                            try:
                                tool_result = tool_executor(tool_name, tool_input)
                                logger.debug(f"Tool result: {json.dumps(tool_result, default=str)[:300]}")
                                
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": json.dumps(tool_result, default=str)
                                })
                            except Exception as tool_error:
                                logger.error(f"Tool execution error: {tool_error}")
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": f"Error executing tool: {str(tool_error)}",
                                    "is_error": True
                                })
                    
                    # Add tool results to message history
                    messages.append({
                        "role": "user",
                        "content": tool_results
                    })
                    
                    # Continue to next iteration to get final response
                    continue
                
                elif stop_reason == 'end_turn':
                    # Extract final text response
                    final_content = ""
                    for content_block in response_body.get('content', []):
                        if content_block.get('type') == 'text':
                            final_content += content_block.get('text', '')
                    
                    logger.info(f"Tool-enabled invocation completed after {iteration} iterations")
                    
                    return {
                        'content': final_content,
                        'stop_reason': stop_reason,
                        'usage': response_body.get('usage', {}),
                        'iterations': iteration,
                        'messages': messages
                    }
                
                else:
                    # Unexpected stop reason
                    logger.warning(f"Unexpected stop reason: {stop_reason}")
                    final_content = ""
                    for content_block in response_body.get('content', []):
                        if content_block.get('type') == 'text':
                            final_content += content_block.get('text', '')
                    
                    return {
                        'content': final_content or "No response generated",
                        'stop_reason': stop_reason,
                        'usage': response_body.get('usage', {}),
                        'iterations': iteration,
                        'messages': messages
                    }
                    
            except Exception as e:
                logger.error(f"Error in tool invocation iteration {iteration}: {e}")
                raise
        
        # Max iterations reached
        logger.warning(f"Max iterations ({max_iterations}) reached in tool-enabled invocation")
        return {
            'content': "Max tool iterations reached without completing the task",
            'stop_reason': 'max_iterations',
            'iterations': iteration,
            'messages': messages
        }
