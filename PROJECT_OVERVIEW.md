# AutoPilot AI - Project Overview

**Team Ragnar** | **AWS AI for Bharat Hackathon 2026**

## Executive Summary

**AutoPilot AI** is a production-grade multi-agent AI SRE system that transforms raw AWS metrics into business-context intelligence with cost analysis in Indian Rupees. It enables autonomous SRE workflows through intelligent recommendations, background job automation, and action tracking.

---

## 🎯 What We Built

### Core Components

1. **Authentication System** - Email/password signup and login with encrypted credential storage
2. **Account Connections** - AWS IAM and GitHub OAuth integration for real account data access
3. **SRE Copilot Chatbot** - Natural language interface for infrastructure queries
4. **Multi-Agent Architecture** - 6 specialized agents (Observability, Cost, CI/CD, Database, Infrastructure, Planner)
5. **Autonomous Jobs** - Scheduled background analysis tasks that run hourly/daily/weekly
6. **Action Center** - Kanban workflow (Suggested → Approved → Executed → Verified) for managing recommendations
7. **Real-Time Alerts** - WebSocket-based live monitoring with severity levels
8. **CI/CD Monitoring** - GitHub Actions integration for build analysis
9. **Dynamic AWS SDK** - LLM-powered autonomous AWS API calls (key innovation)

---

## 🏗️ Architecture

### System Layers

```
┌─ Frontend (React 18 + Vite + Tailwind)
│   ├─ Auth Panel (login/signup)
│   ├─ ChatBot Panel (queries)
│   ├─ Action Center (workflow board)
│   ├─ Autonomous Jobs Panel (scheduling)
│   └─ Live Alerts Panel (WebSocket)
│
├─ Backend (FastAPI with async/await)
│   ├─ REST API endpoints (/api/chat, /api/auth/*)
│   ├─ WebSocket stream (/ws/alerts)
│   ├─ Request routing & validation
│   └─ Credential management
│
├─ Intelligence Layer (AWS Bedrock)
│   ├─ Planner Agent (orchestrates workflows)
│   ├─ Observability Agent (metrics, performance)
│   ├─ Cost Agent (billing, optimization)
│   ├─ CI/CD Agent (build analysis)
│   ├─ Database Agent (query optimization)
│   └─ Infrastructure Agent (security, drift)
│
└─ Execution & Storage
    ├─ AWS SDK Executor (dynamic API calls)
    ├─ GitHub Service (build data)
    ├─ SQLite (auth, connections)
    ├─ JSON Store (actions, jobs, runs)
    └─ Knowledge Base (local or AWS Bedrock)
```

### Data Flow

**User Query → Frontend → FastAPI → Planner Agent → Specialized Agents → AWS APIs + GitHub → Claude Analysis → Structured Response → Frontend**

---

## 🔄 How Everything Integrates

### Authentication Flow
1. User registers/logs in → Password hashed and stored in SQLite
2. Session token generated and stored in browser localStorage
3. All requests include `Authorization: Bearer <token>` header
4. User account stores encrypted AWS credentials and GitHub tokens

### Chat & Analysis Flow
1. User asks question in chatbot
2. System validates AWS connection exists
3. Fetches real AWS context (account ID, costs, available metrics)
4. Routes to appropriate agent(s) based on query intent
5. Agent uses dynamic AWS SDK executor to query services autonomously
6. Claude analyzes results and generates insights
7. Response includes recommendations with cost impact in ₹
8. User can create actions from recommendations

### Action Center Workflow
1. AutoPilot generates recommendations (from chat or autonomous jobs)
2. Suggested actions appear in "Suggested" column
3. Team approves → moves to "Approved"
4. Team executes AWS changes → moves to "Executed"
5. Monitoring verifies success → moves to "Verified"

### Autonomous Jobs
1. User creates job with prompt and schedule (hourly/daily/weekly)
2. Background scheduler registers job with `schedule` library
3. At scheduled time, job executes prompt through agent pipeline
4. Results stored in JSON persistence
5. High-severity findings → auto-create actions
6. User can manually trigger "Run Now"

### Real-Time Alerts
1. Background processes detect issues (CloudWatch, CI/CD, cost anomalies)
2. Alerts created and broadcast via WebSocket
3. LiveAlertsPanel receives in real-time
4. High-severity alerts also create Action Center items
5. User can dismiss alerts or take action

---

## 💡 Key Innovation: Dynamic AWS SDK Execution

**Problem:** Traditional tools hardcode AWS API calls, break when APIs change, can't adapt.

**Solution:** LLM autonomously decides which APIs to call.

```
User: "Show EC2 instances with high CPU"
         ↓
Claude: "Need CloudWatch metrics, let me decide parameters"
         ↓
Tool Call: service=cloudwatch, operation=list_metrics, parameters={...}
         ↓
AWSAPIExecutor: Creates boto3 client → executes operation → returns results
         ↓
Claude: Analyzes results → generates natural language answer
```

**Benefits:**
- No code updates for new AWS services
- Self-correcting (retries on parameter errors)
- Works with any boto3-supported operation
- Future-proof architecture

---

## 🛠️ Tech Stack

### Frontend
- **Framework:** React 18 with Vite bundler
- **Styling:** Tailwind CSS
- **Icons:** Lucide React
- **HTTP:** Axios
- **Real-Time:** Native WebSocket

### Backend
- **Runtime:** Python 3.9+
- **Framework:** FastAPI with Uvicorn (ASGI)
- **Validation:** Pydantic
- **Database:** SQLite (auth) + JSON files (operational state)
- **Threading:** Background scheduler with `schedule` library
- **Encryption:** Fernet (credentials), PBKDF2 (passwords)

### AI & AWS
- **LLM:** AWS Bedrock (Claude Sonnet 3.5)
- **SDK:** Boto3 (dynamic AWS executor)
- **Knowledge Base:** Local file-based or AWS Bedrock KB
- **External APIs:** GitHub REST API

### DevOps
- **Package Manager:** pip (Python), npm (Node)
- **Environment:** Virtual environment + .env configuration
- **Startup:** PowerShell automation script

---

## 📊 Features Breakdown

| Feature | Purpose | Status |
|---------|---------|--------|
| Email/Password Auth | Secure user accounts | ✅ Complete |
| AWS Connection | Real account data access | ✅ Complete |
| GitHub Integration | CI/CD monitoring | ✅ Complete |
| Multi-Agent System | Domain-specific analysis | ✅ Complete |
| Dynamic AWS SDK | Autonomous API execution | ✅ Complete |
| Chatbot Interface | Natural language queries | ✅ Complete |
| Autonomous Jobs | Scheduled analysis | ✅ Complete |
| Action Center | Recommendation workflow | ✅ Complete |
| Real-Time Alerts | WebSocket monitoring | ✅ Complete |
| Cost in ₹ | India-focused pricing | ✅ Complete |
| Role-Based (Future) | Team collaboration | 🔄 Ready for enhancement |

---

## 🎯 Key Differentiators

✅ **Dynamic AWS SDK** - LLM decides APIs, not developers  
✅ **Multi-Agent** - 6 specialized agents + coordinator  
✅ **Autonomous Operations** - Background jobs + action auto-creation  
✅ **India-First** - All costs in ₹ with local pricing  
✅ **End-to-End** - Issue detection → recommendation → action → verification  
✅ **Semantic Intelligence** - Business context, not just metrics  
✅ **No Hardcoding** - Future-proof for AWS service additions  
✅ **Production Ready** - Authentication, encryption, persistence, error handling

---

## 📁 Project Structure

```
AWS Project/
├── frontend/              # React SPA
│   ├── src/components/    # UI components (Auth, Chat, Actions, Jobs, Alerts, etc.)
│   ├── package.json
│   └── vite.config.js
├── agents/                # Specialized AI agents
│   ├── planner_agent.py
│   ├── observability_agent.py
│   ├── cost_agent.py
│   ├── cicd_agent.py
│   ├── db_agent.py
│   └── infra_agent.py
├── services/              # Core services
│   ├── bedrock_client.py     (LLM inference)
│   ├── aws_api_executor.py   (Dynamic AWS SDK)
│   ├── github_service.py     (GitHub integration)
│   ├── scheduler.py          (Background jobs)
│   ├── auth_store.py         (Authentication)
│   ├── ops_state_store.py    (Actions & jobs)
│   └── knowledge_base*.py    (Local or AWS)
├── api/
│   └── routes.py          # REST API endpoints
├── app.py                 # FastAPI application
├── config.py              # Configuration management
├── requirements.txt       # Python dependencies
└── run.ps1               # One-command startup
```

---

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt
cd frontend && npm install && cd ..

# Configure
cp .env.example .env
# Edit .env with AWS credentials and GitHub token

# Run
./run.ps1
# Starts backend on http://localhost:8000
# Starts frontend on http://localhost:5173
```

---

## 🎓 What This Demonstrates

✅ **AWS Bedrock Integration** - Production LLM inference  
✅ **Multi-Agent Architecture** - Coordinated AI workflows  
✅ **Real-World SRE Problem Solving** - Actionable infrastructure intelligence  
✅ **Full-Stack Development** - Frontend, backend, AI integration  
✅ **Security Best Practices** - Encrypted credentials, secure authentication  
✅ **Scalable Design** - Modular agents, pluggable components  
✅ **India-Focused Solutions** - Cost analysis in local currency  
✅ **Production Readiness** - Error handling, persistence, monitoring  

---

**Built for Team Ragnar** | **March 2026**
