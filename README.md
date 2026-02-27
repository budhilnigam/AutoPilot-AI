# AutoPilot AI – Multi-Agent AI SRE System

**Production-grade, multi-agent AI SRE system built on Amazon Bedrock for Indian startups.**

## 🌟 Overview

AutoPilot AI transforms raw DevOps telemetry into semantic insights, providing:

- **Semantic Insights**: Transforms "CPU: 78%" into "Your Celery worker pool is mis-sized relative to Redis throughput causing job starvation. Recommend worker pool = 12. Projected cost reduction = ₹18,300/month."
- **Root Cause Analysis**: Automated incident analysis with confidence scores
- **Cost Optimization**: AWS cost analysis with recommendations in INR
- **Predictive Alerts**: Predict saturation and failures before they occur
- **CI/CD Monitoring**: Build time regression detection
- **Database Optimization**: Query plan analysis and index recommendations
- **Infrastructure Drift**: Automated configuration analysis

## 🏗️ Architecture

### Multi-Agent Design

The system consists of 6 specialized agents coordinated by a Planner Agent:

1. **Planner Agent** (Orchestrator)
   - Routes tasks to specialized agents
   - Synthesizes multi-agent responses
   - Ensures proper execution sequence

2. **Observability Agent**
   - Metric interpretation with business context
   - Statistical anomaly detection (2-sigma)
   - Performance bottleneck attribution

3. **Infrastructure Agent**
   - Docker/ECS configuration analysis
   - Infrastructure drift detection
   - Resource optimization

4. **Database Agent**
   - Query plan analysis
   - Index recommendations
   - Schema optimization

5. **Cost Agent**
   - AWS cost analysis in INR
   - Right-sizing recommendations
   - Cost projections

6. **CI/CD Agent**
   - Build time analysis
   - Regression detection
   - Deployment tracking

### Technology Stack

- **AI/ML**: Amazon Bedrock (Claude 3.5 Sonnet, Claude 3 Haiku)
- **RAG**: Bedrock Knowledge Bases + Titan Embeddings
- **Runtime**: Python 3.11+
- **AWS Services**: CloudWatch, S3, DynamoDB, Cost Explorer, ECS
- **Testing**: Hypothesis (property-based testing)
- **Version Control**: GitHub API integration

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- AWS Account with Bedrock access
- AWS credentials configured (`~/.aws/credentials` or environment variables)

### Installation

```bash
# Clone repository
git clone <repository-url>
cd AutoPilot-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your AWS configuration
```

### Configuration

Edit `.env` file:

```env
AWS_REGION=ap-south-1
BEDROCK_MODEL_ID=anthropic.claude-3-5-sonnet-20241022-v2:0
S3_BUCKET_NAME=your-knowledge-base-bucket
KNOWLEDGE_BASE_ID=your-kb-id
GITHUB_TOKEN=your-github-token
USD_TO_INR_RATE=83.0
```

### Run AutoPilot AI

```bash
python main.py
```

## 📖 Usage Examples

### 1. Natural Language Query

```python
from api.routes import AutoPilotAPI

api = AutoPilotAPI()

result = api.query("What are my current AWS costs and how can I optimize them?")
print(result['insights'])
```

### 2. Metric Analysis

```python
metrics = [
    {
        'metric_name': 'CPUUtilization',
        'metric_type': 'cpu',
        'value': 85.5,
        'unit': 'Percent',
        'dimensions': {'InstanceId': 'i-1234567890abcdef0'}
    }
]

result = api.analyze_metrics(
    metrics=metrics,
    description="EC2 instance performance"
)
```

### 3. Configuration Analysis

```python
dockerfile = """
FROM python:3.11
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "app.py"]
"""

result = api.analyze_configuration(
    config_type='dockerfile',
    config_content=dockerfile
)
```

### 4. Database Query Optimization

```python
result = api.analyze_database_query(
    query_template="SELECT * FROM users WHERE email = ?",
    execution_count=5000,
    avg_duration_ms=150.5,
    tables_accessed=['users']
)
```

### 5. CI/CD Build Analysis

```python
result = api.analyze_build(
    repository="owner/repo"
)
```

### 6. Tool Generation (Agentic Recursion)

```python
result = api.generate_tool(
    tool_type='iam_policy',
    tool_spec={
        'purpose': 'S3 read-only access',
        'permissions': ['s3:GetObject', 's3:ListBucket']
    }
)
```

## 🏛️ Project Structure

```
autopilot-ai/
│
├── agents/                    # AI agents
│   ├── planner_agent.py       # Central orchestrator
│   ├── observability_agent.py # Metrics & anomalies
│   ├── infra_agent.py         # Infrastructure analysis
│   ├── db_agent.py            # Database optimization
│   ├── cost_agent.py          # Cost analysis
│   └── cicd_agent.py          # CI/CD monitoring
│
├── services/                  # AWS integrations
│   ├── bedrock_client.py      # Claude invocation
│   ├── knowledge_base.py      # RAG system
│   ├── tool_generator.py      # Amazon Q integration
│   ├── cloudwatch_client.py   # Metrics & logs
│   ├── billing_client.py      # Cost data
│   └── github_client.py       # Repository data
│
├── models/                    # Data models
│   ├── core_models.py         # Business entities
│   └── agent_protocol.py      # Agent communication
│
├── api/                       # API layer
│   └── routes.py              # REST endpoints
│
├── tests/                     # Tests
│   ├── unit/                  # Unit tests
│   └── property/              # Property-based tests
│
├── main.py                    # Entry point
└── requirements.txt           # Dependencies
```

## 🔒 Core Architectural Constraints

As per the instructions, the system adheres to these non-negotiable constraints:

1. **Exactly 6 Agents**: Observability, Infra, DB, Cost, CICD, Planner
2. **Planner Orchestration**: All inter-agent communication through Planner
3. **RAG Required**: Bedrock Knowledge Bases + Titan Embeddings
4. **Structured Responses**: All outputs parsed to structured JSON
5. **Cost in INR**: All cost outputs in Indian Rupees
6. **Amazon Q Tool Generation**: Dynamic tool generation capability
7. **Property-Based Testing**: 35 correctness properties with Hypothesis

## 🧪 Testing

Run unit tests:

```bash
pytest tests/unit/
```

Run property-based tests:

```bash
pytest tests/property/
```

## 📊 Cost Awareness

All cost outputs are in **INR (₹)** with:
- Monthly projections
- Annual projections
- Confidence scores
- Cost-benefit analysis

Example output:
```json
{
  "summary": "Right-size EC2 instances",
  "cost_impact_inr": -18300,
  "savings_monthly_inr": 18300,
  "savings_annual_inr": 219600
}
```

## 🛡️ Security

- Uses AWS IAM role-based authentication
- No hardcoded credentials
- Follows least privilege principle
- All generated IAM policies validated

## 📈 Monitoring

Logs are written to:
- Standard output (console)
- `autopilot_ai.log` file

Configure log level via `LOG_LEVEL` environment variable.

## 🤝 Contributing

This is a production-grade AI infrastructure system. Follow these guidelines:

- Write modular, typed Python
- Include error handling
- Add unit tests for new features
- Follow existing agent patterns
- Document API changes

## 📝 License

[Add your license here]

## 🙏 Acknowledgments

Built with:
- Amazon Bedrock (Claude 3.5 Sonnet)
- Amazon Bedrock Knowledge Bases
- AWS CloudWatch, S3, Cost Explorer
- GitHub API
- Hypothesis for property-based testing