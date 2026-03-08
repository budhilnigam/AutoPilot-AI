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

## � New Features

### ✨ Modular Knowledge Base
- **Local Mode**: File-based knowledge base for local development (no AWS required)
- **AWS Mode**: Production-ready Bedrock Knowledge Base with S3
- **Easy Toggle**: Switch between modes via config (`USE_LOCAL_KB=true/false`)

### 🎨 Modern Web UI
- **3-Panel Dashboard**:
  - **Left Panel**: Real-time health monitoring for all 6 agents and services
  - **Center Panel**: SRE AI Copilot chatbot with quick-action prompts
  - **Right Panel**: Live alerts from CloudWatch and other sources
- **Tech Stack**: React 18 + Vite + Tailwind CSS + WebSocket

### ⚙️ Centralized Configuration
- Single `config.py` for all backend settings
- Per-agent Bedrock model configuration
- Comprehensive `.env.example` with all variables documented
- Health check interval configuration

### 🔧 AWS SDK Integration
- Bedrock Agents SDK support for tool calling
- Pre-defined tools: CloudWatch metrics, ECS services, RDS instances
- Extensible tool framework for custom integrations

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
- **RAG**: Bedrock Knowledge Bases + Titan Embeddings (or Local KB for dev)
- **Backend**: Python 3.11+ with FastAPI
- **Frontend**: React 18 + Vite + Tailwind CSS
- **AWS Services**: CloudWatch, S3, DynamoDB, Cost Explorer, ECS
- **Testing**: Hypothesis (property-based testing)
- **Real-time**: WebSocket for live alerts

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Node.js 18 or higher
- AWS Account with Bedrock access (optional for local development)
- AWS credentials configured (optional for local development)

### Installation

```bash
# Clone repository
git clone https://github.com/DeepBreach/AutoPilot-AI.git
cd AutoPilot-AI

# Backend setup
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your configuration

# Frontend setup
cd frontend
npm install
cp .env.example .env
cd ..
```

### Configuration

#### For Local Development (No AWS Required)

Edit `.env`:
```env
# Use local knowledge base
USE_LOCAL_KB=true
LOCAL_KB_PATH=./local_kb

# Minimal AWS config (can use dummy values for local testing)
AWS_REGION=us-east-1
BEDROCK_MODEL_ID=openai.gpt-oss-20b-1:0

# Enable CORS for local frontend
ENABLE_CORS=true
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
```

#### For Production (AWS)

Edit `.env`:
```env
# AWS credentials
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=us-east-1

# Use AWS Knowledge Base
USE_LOCAL_KB=false
S3_BUCKET_NAME=your-kb-bucket
KNOWLEDGE_BASE_ID=your-kb-id

# Bedrock models
BEDROCK_MODEL_ID=openai.gpt-oss-20b-1:0

# Optional: Bedrock Agents for tool calling
BEDROCK_AGENT_ID=your-agent-id
BEDROCK_AGENT_ALIAS_ID=your-alias-id
```

### Running

#### Option 1: Quick Start Script (Windows)

```powershell
.\run.ps1
```

#### Option 2: Manual Start

**Terminal 1 - Backend:**
```bash
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

Then open: **http://localhost:5173**

## 📖 Detailed Documentation

- [SETUP.md](SETUP.md) - Comprehensive setup and configuration guide
- [design.md](design.md) - System architecture and design decisions  
- [tasks.md](tasks.md) - Implementation roadmap and task tracking
- [requirements.md](requirements.md) - Detailed requirements specification

## 🎮 Using the Application

### Health Monitoring (Left Panel)
- Auto-refreshes every 5 minutes (configurable)
- Shows status of all 6 AI agents
- Monitors Bedrock, Knowledge Base, CloudWatch, GitHub services
- Visual indicators: Green (healthy), Yellow (degraded), Red (unhealthy)

### AI Copilot Chat (Center Panel)
- Click quick-action prompts for common tasks
- Ask natural language questions about your infrastructure
- Receive structured insights with:
  - Severity levels
  - Business impact analysis
  - Actionable recommendations
  - Cost estimates in INR
  - Confidence scores

### Live Alerts (Right Panel)
- Real-time WebSocket connection
- Filter by severity: Critical, High, Medium, Low
- Shows cost impact and recommendations
- Dismissible alerts
- Commit attribution for issues

## 🔧 Configuration Options

### Health Check Settings
```env
HEALTH_CHECK_INTERVAL_MINUTES=5  # How often to check health
HEALTH_CHECK_TIMEOUT_SECONDS=30  # Timeout for each check
```

### Scheduler Settings
```env
METRIC_CHECK_INTERVAL_MINUTES=5
COST_CHECK_INTERVAL_HOURS=24
BUILD_CHECK_INTERVAL_MINUTES=15
```

### Per-Agent Model Configuration
```env
# Use different models for different agents
PLANNER_AGENT_MODEL_ID=openai.gpt-oss-20b-1:0
OBSERVABILITY_AGENT_MODEL_ID=openai.gpt-oss-20b-1:0
COST_AGENT_MODEL_ID=openai.gpt-oss-20b-1:0  # Cheaper
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/unit/

# Run property-based tests
pytest tests/property/

# Run specific test
pytest tests/unit/test_unit.py::TestObservabilityAgent
```

## 📊 API Endpoints

### Health Checks
- `GET /api/health` - Overall system health
- `GET /api/health/agents` - All 6 agents status
- `GET /api/health/services` - Service status (Bedrock, KB, etc.)

### Chat
- `POST /api/chat` - Send message to SRE AI Copilot

### Prompts
- `GET /api/prompts/default` - Get quick-action prompts

### WebSocket
- `WS /ws/alerts` - Real-time alerts stream

### Other
- `GET /api/config` - Safe configuration summary
- `GET /api/agents` - List all agents

## 🔒 Security Notes

- Never commit `.env` files
- Use IAM roles in production (avoid hardcoded credentials)
- Rotate AWS credentials regularly
- Review agent permissions before deployment
- Enable CloudTrail for audit logging

## 🛠️ Development

### Project Structure
```
autopilot-ai/
├── agents/           # 6 specialized agents
├── api/              # FastAPI routes
├── frontend/         # React + Vite UI
├── models/           # Data models
├── services/         # AWS integrations
├── tests/            # Unit and property tests
├── app.py            # FastAPI application
├── config.py         # Centralized configuration
├── main.py           # CLI entry point
└── run.ps1          # Quick start script
```

### Adding New Features

1. **New Agent**: Create in `agents/` directory
2. **New Tool**: Add to `services/bedrock_agent_service.py`
3. **New UI Panel**: Create in `frontend/src/components/`
4. **New API Endpoint**: Add to `app.py`

## 📈 Roadmap

See [tasks.md](tasks.md) for detailed implementation status.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

[Add your license here]

## 🙏 Acknowledgments

- Amazon Bedrock team for the AI infrastructure
- Indian startup community for feedback and requirements
- Open-source contributors

---

**Built with ❤️ for the Indian startup ecosystem**

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