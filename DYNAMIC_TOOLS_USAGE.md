# Dynamic AWS SDK Tool Usage

## Overview

Instead of hardcoding AWS API calls throughout the codebase, this system uses **dynamic tool execution** where the LLM (Claude via Bedrock) decides:
1. Which AWS service to call
2. Which operation to use
3. What parameters to pass

This eliminates:
- ❌ Hardcoded API calls with wrong parameters (like the `MaxRecords` error)
- ❌ Need to update code for every new AWS API
- ❌ Parameter validation errors from outdated knowledge

## Architecture

```
User Query
    ↓
LLM (Claude via Bedrock)
    ↓
Tool Request: aws_api_executor
    ↓
AWSAPIExecutor.execute(service, operation, parameters)
    ↓
Boto3 Dynamic Client
    ↓
AWS API
    ↓
Result → LLM → Final Answer to User
```

## Components

### 1. AWSAPIExecutor (`services/aws_api_executor.py`)

Generic AWS SDK executor that can call any boto3 operation:

```python
from services.aws_api_executor import get_aws_executor

executor = get_aws_executor()

# The LLM decides to make this call
result = executor.execute(
    service='cloudwatch',
    operation='list_metrics',
    parameters={
        'Namespace': 'AWS/EC2',
        'MetricName': 'CPUUtilization'
    }
)
```

### 2. Tool-Enabled Bedrock Client (`services/bedrock_client.py`)

Enhanced to support Anthropic's tool use (function calling):

```python
def execute_tool(tool_name, tool_input):
    if tool_name == "aws_api_executor":
        return executor.execute(**tool_input)

response = bedrock_client.invoke_with_tools(
    system_prompt="You are an AWS expert...",
    user_prompt="List all EC2 instances",
    tools=[executor.get_tool_definition()],
    tool_executor=execute_tool
)
```

### 3. Dynamic Agent Methods

Agents now have methods that use dynamic tools instead of hardcoded calls.

**Example:** `ObservabilityAgent.analyze_with_dynamic_tools()`

## Usage Examples

### Example 1: Observability Query

```python
from agents.observability_agent import ObservabilityAgent

agent = ObservabilityAgent()

# LLM figures out which AWS APIs to call
response = agent.analyze_with_dynamic_tools(
    task_id="task-123",
    user_query="Show me EC2 instances with high CPU usage in the last hour",
    context={'aws_account': {...}}
)

print(response.data['full_response'])
```

**What happens behind the scenes:**
1. LLM receives the query
2. LLM decides to call: `cloudwatch.list_metrics()` to find CPU metrics
3. LLM calls: `cloudwatch.get_metric_statistics()` to get actual data
4. LLM analyzes results and returns insights

### Example 2: Direct Executor Usage

```python
from services.aws_api_executor import get_aws_executor

executor = get_aws_executor()

# List all S3 buckets
buckets = executor.execute(
    service='s3',
    operation='list_buckets',
    parameters={}
)

# Get database instances
dbs = executor.execute(
    service='rds',
    operation='describe_db_instances',
    parameters={}
)

# Get monthly costs
costs = executor.execute(
    service='ce',
    operation='get_cost_and_usage',
    parameters={
        'TimePeriod': {
            'Start': '2026-03-01',
            'End': '2026-03-08'
        },
        'Granularity': 'MONTHLY',
        'Metrics': ['UnblendedCost']
    }
)
```

### Example 3: Custom Agent Tool Integration

```python
from services.bedrock_client import BedrockClient
from services.aws_api_executor import get_aws_executor

bedrock = BedrockClient()
executor = get_aws_executor()

def my_tool_executor(tool_name, tool_input):
    """Handle tool execution requests from LLM"""
    if tool_name == "aws_api_executor":
        return executor.execute(**tool_input)
    return {"error": "Unknown tool"}

# Let LLM query AWS dynamically
result = bedrock.invoke_with_tools(
    system_prompt="You are helping analyze AWS infrastructure.",
    user_prompt="Find all Lambda functions that failed in the last 24 hours",
    tools=[executor.get_tool_definition()],
    tool_executor=my_tool_executor,
    max_iterations=10
)

print(result['content'])
```

## Benefits

### 1. **Self-Correcting**
If the LLM makes a mistake (like using `MaxRecords` for `list_metrics`), it gets an error and tries again with correct parameters.

### 2. **No Code Updates Required**
AWS releases new APIs → LLM can use them immediately via boto3, no code changes needed.

### 3. **Natural Language Interface**
Users ask questions in plain English, LLM figures out the AWS calls.

### 4. **Audit Trail**
All AWS API calls are logged with parameters for debugging and compliance.

### 5. **Error Handling**
Parameter validation errors are caught and returned to the LLM to fix.

## Comparison: Old vs New

### Old Approach (Hardcoded)
```python
# app.py - BEFORE
cloudwatch_client = boto3.client('cloudwatch')
metrics_response = cloudwatch_client.list_metrics(
    Namespace='AWS/EC2',
    MetricName='CPUUtilization',
    MaxRecords=20,  # ❌ WRONG! Not a valid parameter
)
```

**Problems:**
- Developers must know exact parameter names
- Static code, hard to extend
- Errors like "Unknown parameter: MaxRecords"

### New Approach (Dynamic)
```python
# agents/observability_agent.py - AFTER
response = agent.analyze_with_dynamic_tools(
    task_id="task-123",
    user_query="Show me EC2 CPU metrics",
    context={}
)
```

**Advantages:**
- LLM knows AWS SDK documentation
- Dynamic, adapts to query
- Self-correcting on errors

## Tool Definition

The LLM sees this tool schema:

```json
{
  "name": "aws_api_executor",
  "description": "Execute AWS SDK operations dynamically...",
  "input_schema": {
    "type": "object",
    "properties": {
      "service": {
        "type": "string",
        "description": "AWS service name (e.g., 'ec2', 's3', 'cloudwatch')"
      },
      "operation": {
        "type": "string",
        "description": "API operation name (e.g., 'describe_instances')"
      },
      "parameters": {
        "type": "object",
        "description": "Parameters for the operation"
      }
    },
    "required": ["service", "operation"]
  }
}
```

## Available AWS Services

The executor supports all boto3 services:
- `ec2` - EC2 instances
- `s3` - S3 storage
- `cloudwatch` - Metrics and monitoring
- `logs` - CloudWatch Logs
- `ce` - Cost Explorer
- `rds` - Relational databases
- `dynamodb` - NoSQL database
- `lambda` - Lambda functions
- `ecs` - Containers
- `iam` - Identity management
- And 100+ more!

## Migration Path

### Phase 1: Fix Immediate Issues ✅
- Fixed the `MaxRecords` error in `app.py`
- Created `AWSAPIExecutor` service
- Added tool support to `BedrockClient`

### Phase 2: Agent Updates (In Progress)
- ✅ Updated `ObservabilityAgent` with `analyze_with_dynamic_tools()`
- ⏳ Update other agents (Cost, Infrastructure, DB, CI/CD)
- ⏳ Update `PlannerAgent` to route to dynamic methods

### Phase 3: Replace Legacy Calls
- Replace hardcoded AWS calls in `app.py`
- Remove static CloudWatch client usage
- Use dynamic executor everywhere

### Phase 4: Expand Capabilities
- Add more tools (GitHub, Slack, PagerDuty)
- Multi-tool orchestration
- Complex workflows

## Testing

### Test the Executor Directly

```python
# test_dynamic_tools.py
from services.aws_api_executor import get_aws_executor

executor = get_aws_executor()

# Test CloudWatch
try:
    result = executor.execute(
        service='cloudwatch',
        operation='list_metrics',
        parameters={'Namespace': 'AWS/EC2'}
    )
    print(f"Found {len(result.get('Metrics', []))} metrics")
except Exception as e:
    print(f"Error: {e}")
```

### Test Through Agent

```python
# Test observability agent
from agents.observability_agent import ObservabilityAgent

agent = ObservabilityAgent()

response = agent.analyze_with_dynamic_tools(
    task_id="test-1",
    user_query="What are the current CloudWatch namespaces available?"
)

print(response.data['full_response'])
```

## Best Practices

### 1. **Let the LLM Decide**
Don't hardcode AWS calls. Let the LLM figure out what's needed:
```python
# ❌ BAD
result = cloudwatch.list_metrics(Namespace='AWS/EC2')

# ✅ GOOD
response = agent.analyze_with_dynamic_tools(
    user_query="Show me all CloudWatch metrics"
)
```

### 2. **Provide Context**
Give the LLM AWS account context:
```python
context = {
    'aws_account': {
        'account_id': '123456789',
        'region': 'us-east-1',
        'monthly_cost': 150.50
    }
}
response = agent.analyze_with_dynamic_tools(query, context)
```

### 3. **Handle Errors Gracefully**
The LLM will get AWS errors and can retry:
```python
# If LLM uses wrong parameter, it sees:
# "ParamValidationError: Unknown parameter: MaxRecords"
# Then it retries with correct parameters
```

### 4. **Log Everything**
The executor logs all AWS calls for debugging:
```
INFO: Executing AWS API: cloudwatch.list_metrics with params: {...}
INFO: API call successful: cloudwatch.list_metrics
```

## Troubleshooting

### Error: "Service 'xyz' does not have operation 'abc'"
- Check boto3 documentation for correct operation name
- Use snake_case: `describe_instances` not `DescribeInstances`

### Error: "Parameter validation failed"
- Let the LLM see the error - it will retry with correct parameters
- Check AWS SDK docs for valid parameters

### Tool not being called
- Check system prompt mentions the tool
- Ensure tool definition is passed to `invoke_with_tools()`
- Increase `max_iterations` if needed

## Future Enhancements

1. **Multi-Tool Support**: Add GitHub, Datadog, Slack tools
2. **Caching**: Cache frequent AWS API results
3. **Cost Optimization**: Track API call costs
4. **Rate Limiting**: Prevent excessive AWS API calls
5. **Parallel Execution**: Execute multiple AWS calls in parallel

## Conclusion

The dynamic AWS SDK tool approach transforms this from a **static codebase** requiring updates for every AWS API into a **self-evolving system** where the LLM decides how to interact with AWS based on user queries.

**Key Insight:** AWS SDK already contains all the APIs. We just need to let the AI use them dynamically rather than hardcoding calls.
