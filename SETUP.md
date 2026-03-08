# AutoPilot AI - Setup and Run Instructions

## Quick Start

### 1. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Copy environment configuration
cp .env.example .env

# Edit .env with your AWS credentials and settings
# At minimum, configure:
# - AWS_ACCESS_KEY_ID
# - AWS_SECRET_ACCESS_KEY
# - AWS_REGION
# - BEDROCK_MODEL_ID

# For local development, ensure USE_LOCAL_KB=true
```

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install Node.js dependencies
npm install

# Copy frontend environment configuration
cp .env.example .env

# Ensure VITE_API_URL points to your backend (default: http://localhost:8000)
```

### 3. Running the Application

#### Option A: Run Backend and Frontend Separately

**Terminal 1 - Backend:**
```bash
# From project root
python -m uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

**Terminal 2 - Frontend:**
```bash
# From frontend directory
cd frontend
npm run dev
```

Access the application at: http://localhost:5173

#### Option B: Run Everything with Script

```bash
# Windows PowerShell
.\run.ps1

# Linux/Mac
./run.sh
```

## Architecture Overview

### Backend (Python + FastAPI)
- **FastAPI**: REST API and WebSocket server
- **6 AI Agents**: Planner, Observability, Infrastructure, Database, Cost, CI/CD
- **AWS Services**: Bedrock, CloudWatch, ECS, RDS, Cost Explorer
- **Knowledge Base**: Switchable between local (dev) and AWS (prod)

### Frontend (React + Vite)
- **3-Panel Layout**:
  - Left: Health monitoring for agents and services
  - Center: SRE AI Copilot chatbot
  - Right: Live alerts from CloudWatch
- **Tech Stack**: React 18, Tailwind CSS, WebSocket for real-time updates

## Configuration

### Backend Configuration (`.env`)

Key settings:
```env
# Use local KB for development
USE_LOCAL_KB=true
LOCAL_KB_PATH=./local_kb

# Health check interval (minutes)
HEALTH_CHECK_INTERVAL_MINUTES=5

# API settings
API_PORT=8000
ENABLE_CORS=true
```

### Frontend Configuration (`frontend/.env`)

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
```

## Key Features

### 1. Local Knowledge Base (Development Mode)
- File-based storage for quick prototyping
- No AWS credentials required for basic testing
- Automatically creates `./local_kb` directory

### 2. Modular Configuration
- Single `config.py` for all backend settings
- Per-agent model configuration
- Easy switching between local and AWS modes

### 3. Health Monitoring
- Auto-refresh health checks (configurable interval)
- Real-time status for all 6 agents
- Service health (Bedrock, KB, CloudWatch, GitHub)

### 4. AI-Powered Chat
- Default prompts for quick actions
- Multi-agent orchestration via Planner
- Structured responses with insights and recommendations

### 5. Live Alerts
- WebSocket-based real-time alerts
- Severity filtering (Critical, High, Medium, Low)
- Cost impact and recommendations

## Switching to Production (AWS Mode)

1. Update `.env`:
```env
USE_LOCAL_KB=false
KNOWLEDGE_BASE_ID=your-actual-kb-id
S3_BUCKET_NAME=your-kb-bucket
```

2. Ensure AWS credentials are configured:
```env
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
```

3. Optional: Configure Bedrock Agents for tool calling:
```env
BEDROCK_AGENT_ID=your-agent-id
BEDROCK_AGENT_ALIAS_ID=your-alias-id
```

## Troubleshooting

### Backend won't start
- Check Python version (3.11+ required)
- Verify all dependencies: `pip install -r requirements.txt`
- Check `.env` file exists and is configured

### Frontend won't start
- Check Node.js version (18+ required)
- Clear node_modules: `rm -rf node_modules && npm install`
- Verify backend is running on port 8000

### Health checks show "unhealthy"
- For local development with `USE_LOCAL_KB=true`, some AWS services may show degraded (expected)
- Ensure AWS credentials are valid if using AWS services
- Check CloudWatch permissions if using metrics

### No alerts appearing
- Check WebSocket connection status (green dot in Alerts panel)
- Verify backend WebSocket endpoint is accessible
- Check browser console for WebSocket errors

## Development Tips

### Testing Agents Locally
```python
# Test without AWS credentials
from agents.observability_agent import ObservabilityAgent
from models.core_models import MetricData, MetricType

agent = ObservabilityAgent()
# Agent will work with mock/local data
```

### Adding New Default Prompts
Edit `app.py` and add to the `/api/prompts/default` endpoint.

### Customizing Health Check Interval
Edit `.env`:
```env
HEALTH_CHECK_INTERVAL_MINUTES=10  # Change from 5 to 10 minutes
```

## Next Steps

1. **Configure AWS Services**: Set up Bedrock Knowledge Base, CloudWatch, etc.
2. **Add Real Data**: Connect to your actual infrastructure
3. **Customize Agents**: Modify agent behavior in `agents/` directory
4. **Extend Frontend**: Add new panels or features in `frontend/src/components/`
5. **Deploy**: Use Docker, ECS, or your preferred deployment method

## Support

For issues or questions:
- Check the logs: `autopilot_ai.log`
- Review agent responses in the chat
- Inspect health check panel for service status
