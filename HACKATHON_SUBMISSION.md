# AutoPilot AI - Complete Feature Documentation

**Team Ragnar** | **AWS AI for Bharat Hackathon 2026**

---

## 🌟 Executive Summary

**AutoPilot AI** is a production-grade multi-agent AI SRE (Site Reliability Engineering) system designed specifically for Indian startups. Unlike traditional monitoring tools that show raw metrics, AutoPilot AI provides **semantic intelligence** - understanding the business context behind every metric, delivering actionable insights, and automating SRE workflows.

### Key Innovation: Dynamic AWS SDK Tool Calling

Instead of hardcoded API calls, AutoPilot AI uses **Claude Sonnet via AWS Bedrock** to dynamically decide which AWS services to query, which operations to perform, and what parameters to use - all based on natural language queries. This eliminates static rules and enables truly adaptive infrastructure intelligence.

### India-First Design

- All costs displayed in **₹ (Indian Rupees)**
- ROI calculations with Indian cloud pricing
- Built for startups scaling in the Indian market
- Cost optimization specifically for AWS India regions

---

## 📋 Table of Contents

1. [System Architecture](#system-architecture)
2. [Authentication & User Management](#authentication--user-management)
3. [Account Connections (AWS & GitHub)](#account-connections-aws--github)
4. [SRE Copilot Chatbot](#sre-copilot-chatbot)
5. [Multi-Agent System](#multi-agent-system)
6. [CI/CD Workflow Monitoring](#cicd-workflow-monitoring)
7. [Autonomous Jobs](#autonomous-jobs)
8. [Action Center](#action-center)
9. [Real-Time Monitoring & Alerts](#real-time-monitoring--alerts)
10. [Knowledge Base System](#knowledge-base-system)
11. [Dynamic AWS SDK Execution](#dynamic-aws-sdk-execution)
12. [Technical Stack](#technical-stack)

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Web Frontend (React)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   Auth   │  │  Copilot │  │  Actions │  │   Jobs   │       │
│  │  Panel   │  │   Chat   │  │  Center  │  │  Panel   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              ↕ REST API / WebSocket
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Backend Server                         │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              AutoPilot API (Orchestration)             │    │
│  └────────────────────────────────────────────────────────┘    │
│                              ↕                                   │
│  ┌────────────────────────────────────────────────────────┐    │
│  │              Planner Agent (Coordinator)               │    │
│  └────────────────────────────────────────────────────────┘    │
│                              ↕                                   │
│  ┌──────────┬──────────┬──────────┬──────────┬──────────┐     │
│  │   Obs    │   Cost   │  CI/CD   │   DB     │  Infra   │     │
│  │  Agent   │  Agent   │  Agent   │  Agent   │  Agent   │     │
│  └──────────┴──────────┴──────────┴──────────┴──────────┘     │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                      External Services                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │   AWS    │  │  GitHub  │  │ Bedrock  │  │Knowledge │       │
│  │   APIs   │  │   API    │  │  Claude  │  │   Base   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

### Architecture Highlights

- **Modern Frontend**: React 18 + Vite + Tailwind CSS
- **High-Performance Backend**: FastAPI with async/await
- **AI-Powered Intelligence**: AWS Bedrock with Claude Sonnet
- **Persistent Storage**: SQLite for auth + JSON for operational state
- **Real-Time Updates**: WebSocket for live alerts and notifications
- **Modular Design**: Pluggable knowledge base (local or AWS)

---

## 🔐 Authentication & User Management

### Overview

Secure authentication system with email/password registration, session management, and account connection tracking.

### User Flow

#### 1. **Registration (Sign Up)**

```
User → Frontend → POST /api/auth/register → Backend
                                           ↓
                                    SQLite Database
                                           ↓
                                    User Created
                                           ↓
                                    Session Token
                                           ↓
                                    Frontend ← Response
```

**Process:**
1. User enters email and password (minimum 8 characters)
2. Backend hashes password with unique salt using PBKDF2
3. User record created in SQLite database
4. Session token generated (cryptographically secure random)
5. Token returned to frontend and stored in localStorage

**Security Features:**
- Password hashing with PBKDF2-HMAC-SHA256
- Unique salt per user
- Session tokens with expiration (30 days default)
- SQL injection protection via parameterized queries

#### 2. **Login (Sign In)**

```
User → Frontend → POST /api/auth/login → Backend
                                        ↓
                                 Verify Password
                                        ↓
                                 Create Session
                                        ↓
                                 Return Token
                                        ↓
                                Frontend ← Response
```

**Process:**
1. User enters credentials
2. Backend retrieves user by email
3. Password verified against stored hash
4. New session token generated
5. Token stored in database and returned to user

#### 3. **Authentication Persistence**

- Session tokens stored in browser localStorage
- All API requests include `Authorization: Bearer <token>` header
- Backend validates token on every protected route
- Invalid/expired tokens return 401 Unauthorized

### UI Components

**AuthPanel Component:**
- Clean, modern login/signup interface
- Real-time form validation
- Toggle between login and registration modes
- Error display with user-friendly messages
- Gradient background with animated entry

### Database Schema

```sql
-- Users table
CREATE TABLE users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    password_salt TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- Sessions table
CREATE TABLE sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    revoked_at TEXT,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### API Endpoints

| Endpoint | Method | Description | Authentication |
|----------|--------|-------------|----------------|
| `/api/auth/register` | POST | Create new user account | No |
| `/api/auth/login` | POST | Login with credentials | No |
| `/api/auth/logout` | POST | Revoke session token | Required |
| `/api/auth/me` | GET | Get current user info | Required |

---

## 🔗 Account Connections (AWS & GitHub)

### Overview

AutoPilot AI requires connection to both AWS and GitHub to provide comprehensive analysis. AWS connection is **mandatory** for core functionality, while GitHub connection enables CI/CD monitoring features.

### AWS Connection Flow

#### Prerequisites

Users need AWS credentials with specific IAM permissions. The system provides a **complete IAM policy** that can be copied directly into AWS.

#### Connection Process

```
User → Account Settings → Enter AWS Credentials → POST /api/auth/connect/aws
                                                          ↓
                                                  Validate Credentials
                                                          ↓
                                                  Test Permissions
                                                          ↓
                                                  Encrypt & Store
                                                          ↓
                                                  Return Status
```

**Step-by-Step:**

1. **User enters credentials:**
   - AWS Access Key ID
   - AWS Secret Access Key
   - Session Token (optional, for temporary credentials)
   - AWS Region (default: us-east-1)

2. **Backend validation:**
   - Attempts `sts:GetCallerIdentity` to verify credentials
   - Tests permissions for key services (CloudWatch, EC2, RDS, etc.)
   - Identifies missing permissions

3. **Secure storage:**
   - Credentials encrypted using Fernet (symmetric encryption)
   - Encryption key stored separately from database
   - Stored in SQLite with foreign key to user

4. **Connection status:**
   - Returns list of missing permissions (if any)
   - Connection marked as active
   - Frontend displays connection status

#### Required AWS Permissions

The system requires these AWS permissions (full policy provided in UI):

**Core Services:**
- `sts:GetCallerIdentity` - Identity verification
- `cloudwatch:*` - Metrics and alarms
- `logs:*` - Log analysis
- `ce:*` - Cost Explorer for billing data

**Compute & Database:**
- `ec2:Describe*` - EC2 instance data
- `rds:Describe*` - Database monitoring
- `lambda:List*`, `lambda:Get*` - Serverless functions
- `ecs:*` - Container orchestration

**Storage & Networking:**
- `s3:List*`, `s3:Get*`, `s3:Put*` - S3 operations
- `elasticloadbalancing:Describe*` - Load balancer data

**AI Services:**
- `bedrock:InvokeModel` - LLM inference
- `bedrock:Retrieve` - Knowledge base queries

### GitHub Connection Flow

#### Two Methods: OAuth & Personal Access Token

#### Method 1: OAuth (Recommended)

```
User → Click "Connect GitHub" → OAuth Popup Opens
                                       ↓
                                  GitHub Auth
                                       ↓
                              User Grants Access
                                       ↓
                         Callback with Auth Code
                                       ↓
                         Exchange for Access Token
                                       ↓
                              Store Token
                                       ↓
                          Connection Complete
```

**OAuth Flow:**
1. User clicks "Connect with GitHub"
2. Popup window opens to GitHub OAuth
3. User authorizes access (read:repo, workflow, read:org)
4. GitHub redirects to callback URL with auth code
5. Backend exchanges code for access token
6. Token encrypted and stored
7. Repository owner/name saved for monitoring

#### Method 2: Personal Access Token

```
User → Settings → Enter PAT + Repo Info → POST /api/auth/connect/github
                                                    ↓
                                            Validate Token
                                                    ↓
                                            Test Repo Access
                                                    ↓
                                            Encrypt & Store
```

**PAT Setup:**
1. User generates GitHub Personal Access Token
2. Required scopes: `repo`, `workflow`, `read:org`
3. User enters token, repository owner, and repository name
4. Backend validates token by querying GitHub API
5. Token encrypted and stored with repository info

### Connection Management

**AccountConnections Component:**

Features:
- Real-time connection status indicators
- Copy-to-clipboard IAM policy
- Expandable sections for ease of use
- Test connection functionality
- Disconnect option for both services
- Missing permissions warnings

**Connection States:**

| Service | Status | Icon | Description |
|---------|--------|------|-------------|
| AWS | Connected | ✅ Green | All permissions granted |
| AWS | Partial | ⚠️ Yellow | Missing some permissions |
| AWS | Disconnected | ❌ Red | Not connected |
| GitHub | Connected | ✅ Green | OAuth or PAT valid |
| GitHub | Disconnected | ❌ Red | Not connected |

### Runtime Connection Validation

**Every chat request:**
1. Validates current user session
2. Checks AWS connection exists
3. Fetches real-time AWS context:
   - Account identity (account ID, user ARN)
   - Current month's cost in USD → ₹
   - Available CloudWatch metrics
   - Active services
4. Optionally checks GitHub connection (if needed)
5. Passes context to agent for grounded analysis

**Connection Enforcement:**
- `/api/chat` endpoint requires AWS connection
- Returns 403 error if AWS not connected
- GitHub optional - features disabled if not connected
- Clear error messages guide users to connect accounts

### Database Schema

```sql
-- AWS Connections
CREATE TABLE aws_connections (
    user_id INTEGER PRIMARY KEY,
    access_key_id_encrypted BLOB NOT NULL,
    secret_access_key_encrypted BLOB NOT NULL,
    session_token_encrypted BLOB,
    region TEXT NOT NULL,
    connected_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- GitHub Connections
CREATE TABLE github_connections (
    user_id INTEGER PRIMARY KEY,
    access_token_encrypted BLOB NOT NULL,
    repo_owner TEXT NOT NULL,
    repo_name TEXT NOT NULL,
    connected_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(id)
);
```

### Security Measures

**Encryption:**
- All credentials encrypted with Fernet (AES-128-CBC + HMAC)
- Unique encryption key per installation
- Keys never transmitted, only stored locally
- Database stores only encrypted values

**Access Control:**
- Session tokens required for all connection endpoints
- Users can only access their own connections
- Tokens validated on every request
- Sessions expire after 30 days

---

## 💬 SRE Copilot Chatbot

### Overview

The core interface for interacting with AutoPilot AI. Users ask questions in natural language, and the system provides intelligent, context-aware answers grounded in their actual AWS infrastructure.

### User Experience

#### Chat Interface (ChatbotPanel Component)

**Features:**
- Clean, conversational UI
- Message history with scroll
- Auto-scroll to latest message
- Loading indicators during analysis
- Syntax-highlighted code blocks
- Markdown rendering for rich responses

#### Quick-Action Prompts

Pre-configured prompts for common SRE tasks:

1. **"Analyze my AWS infrastructure for performance issues"**
2. **"Show me cost optimization opportunities"**
3. **"Detect any infrastructure drift or anomalies"**
4. **"Analyze recent build failures and CI/CD trends"**
5. **"Show database performance and suggest optimizations"**
6. **"Predict resource saturation in the next 7 days"**
7. **"Generate incident response recommendations"**
8. **"Analyze CloudWatch alarms and alert patterns"**

Users can click these prompts to instantly trigger analysis.

### Chat Request Flow

```
User Message → Frontend
                  ↓
        POST /api/chat {message}
                  ↓
          Validate Session
                  ↓
       Check AWS Connection
                  ↓
      Build AWS Runtime Context
                  ↓
         Route to Planner Agent
                  ↓
      Planner Analyzes Intent
                  ↓
   Routes to Specialized Agent(s)
                  ↓
    Agents Execute with Tools
                  ↓
       Results Synthesized
                  ↓
      Response with Insights
                  ↓
           Frontend ← JSON
```

### Request Processing

#### 1. **Context Building**

Before any query, the system gathers:

```python
{
    "account_identity": {
        "account_id": "123456789012",
        "user_arn": "arn:aws:iam::123456789012:user/autopilot",
        "user_id": "AIDAI..."
    },
    "current_costs_inr": "₹45,230",
    "current_costs_usd": "$542.00",
    "available_metrics": [
        {"namespace": "AWS/EC2", "metric": "CPUUtilization"},
        {"namespace": "AWS/RDS", "metric": "DatabaseConnections"}
    ],
    "services_in_use": ["EC2", "RDS", "Lambda", "S3"],
    "region": "us-east-1"
}
```

This ensures every answer is **grounded in reality**, not generic LLM responses.

#### 2. **Intent Analysis**

Planner Agent examines the query and determines:
- Which specialized agents to invoke
- Sequence of operations
- Required AWS services
- Expected output format

#### 3. **Agent Execution**

Specialized agents execute with:
- User's AWS credentials (decrypted on-the-fly)
- GitHub connection (if relevant)
- Access to dynamic AWS SDK executor
- Knowledge base for best practices

#### 4. **Response Assembly**

The system returns:

```json
{
    "response": "Full text response with analysis",
    "thinking": "Internal reasoning (if enabled)",
    "insights": [
        {
            "type": "CRITICAL",
            "title": "Worker Pool Undersized",
            "description": "Your Celery workers can't keep up...",
            "severity": "CRITICAL",
            "confidence": 0.95,
            "cost_impact_inr": "₹16,308/month savings",
            "agent": "observability"
        }
    ],
    "recommendations": [
        {
            "title": "Scale ECS Service",
            "action": "Increase task count from 2 to 4",
            "impact": "Reduces queue backlog by 85%",
            "risk": "Low",
            "implementation_time": "5 minutes",
            "script": "aws ecs update-service --cluster production --service api-service --desired-count 4"
        }
    ],
    "agent_type": "observability",
    "execution_time_ms": 3542,
    "aws_context": {...}
}
```

### Response Rendering

#### Insights Display

Each insight shows:
- **Severity Badge**: Critical/High/Medium/Low with color coding
- **Title**: Brief description
- **Details**: Full explanation with business context
- **Confidence Score**: AI confidence percentage
- **Cost Impact**: Financial impact in ₹
- **Source Agent**: Which agent generated the insight

#### Recommendations Display

Each recommendation includes:
- **Action Title**: What to do
- **Impact Statement**: Expected results
- **Risk Assessment**: Implementation risk level
- **Time Estimate**: How long it takes
- **Migration Script**: Copy-paste ready commands
- **ROI Calculation**: Cost-benefit analysis

### Example Interaction

**User:** "Analyze my ECS services for performance issues"

**System Processing:**
1. Fetches user's AWS credentials
2. Builds runtime context (account info, costs)
3. Planner routes to Observability Agent
4. Agent uses dynamic tools to:
   - List ECS clusters
   - Describe services
   - Fetch CloudWatch CPU/Memory metrics
   - Check task health
5. Claude analyzes data and generates insights

**Response:**

```
🔴 CRITICAL: Worker Pool Undersized

ROOT CAUSE:
Your Celery worker pool (2 tasks) can't keep up with Redis queue depth
(1,247 jobs). Average job processing is delayed by 12 seconds, causing
job starvation.

BUSINESS IMPACT:
- User requests timing out
- Background jobs backing up
- Potential data processing delays

RECOMMENDATION:
Scale ECS service from 2 to 4 tasks

COST ANALYSIS:
├─ Current Cost: ₹1,992/month
├─ Proposed Cost: ₹3,984/month
├─ Additional Cost: +₹1,992/month
└─ Savings: ₹18,300/month (prevents Redis over-provisioning)

NET SAVINGS: ₹16,308/month (818% ROI)
RISK: Low (horizontal scaling is safe)
IMPLEMENTATION: 5 minutes

MIGRATION SCRIPT:
aws ecs update-service \
  --cluster production \
  --service api-service \
  --desired-count 4
```

### Chat API Endpoint

```
POST /api/chat
Authorization: Bearer <token>

Request:
{
    "message": "Analyze my database performance"
}

Response:
{
    "response": "...",
    "insights": [...],
    "recommendations": [...],
    "agent_type": "database",
    "execution_time_ms": 2834,
    "aws_context": {...}
}
```

---

## 🤖 Multi-Agent System

### Overview

AutoPilot AI uses a **coordinated multi-agent architecture** where specialized agents handle different SRE domains, all orchestrated by a central Planner Agent.

### Agent Hierarchy

```
                    ┌─────────────────┐
                    │  Planner Agent  │
                    │  (Coordinator)  │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
    ┌────▼────┐         ┌────▼────┐        ┌────▼────┐
    │  Agent  │         │  Agent  │        │  Agent  │
    │    1    │         │    2    │   ...  │    6    │
    └─────────┘         └─────────┘        └─────────┘
```

### Specialized Agents

#### 1. **Planner Agent** (Orchestrator)

**Responsibilities:**
- Analyze user queries
- Determine intent and task decomposition
- Route tasks to appropriate specialized agents
- Coordinate multi-agent workflows
- Synthesize responses from multiple agents
- Ensure consistent output format

**Capabilities:**
- Natural language understanding
- Task decomposition
- Agent selection logic
- Response aggregation
- Error handling and fallback

**Example Routing:**

| Query | Routed To |
|-------|-----------|
| "Why are my costs high?" | Cost Agent |
| "Database is slow" | Database Agent |
| "Analyze build failures" | CI/CD Agent |
| "Overall infrastructure health" | Multiple agents |

#### 2. **Observability Agent**

**Domain:** Infrastructure monitoring, metrics analysis, alerting

**Capabilities:**
- CloudWatch metrics analysis
- CPU, memory, disk, network utilization
- Auto-scaling evaluation
- Alert pattern recognition
- Anomaly detection
- Performance regression analysis

**Keywords:** performance, monitoring, metrics, alerts, cpu, memory, latency, throughput

**Tools Used:**
- `cloudwatch:GetMetricData`
- `cloudwatch:DescribeAlarms`
- `ec2:DescribeInstances`
- `ecs:DescribeServices`
- `lambda:GetFunction`

**Sample Output:**
```
⚠️ HIGH: Elevated CPU Usage Detected

ECS Service 'api-service' averaging 87% CPU over last 6 hours.
Recommend scaling from 2 to 3 tasks.

Cost Impact: +₹996/month
Performance Gain: 42% latency reduction
```

#### 3. **Cost Agent**

**Domain:** AWS billing, cost optimization, budget management

**Capabilities:**
- Cost anomaly detection
- Reserved instance recommendations
- Rightsizing analysis (CPU, memory, storage)
- Savings plan evaluation
- Cost forecasting
- Multi-service cost attribution

**Keywords:** cost, billing, pricing, savings, budget, spend, expensive

**Tools Used:**
- `ce:GetCostAndUsage`
- `ce:GetCostForecast`
- `ec2:DescribeInstances` (for rightsizing)
- `rds:DescribeDBInstances`

**Currency Conversion:**
All costs automatically converted to ₹ using real-time exchange rates.

**Sample Output:**
```
💰 COST ANOMALY: RDS Over-Provisioned

db.r5.2xlarge running at 27% utilization
Downgrade to db.r5.xlarge

SAVINGS: ₹20,750/month (50% reduction)
ANNUAL IMPACT: ₹2.49L/year
```

#### 4. **CI/CD Agent**

**Domain:** Build monitoring, deployment tracking, GitHub integration

**Capabilities:**
- Build time trend analysis
- Failure pattern detection
- Test flakiness identification
- Deployment frequency tracking
- GitHub Actions workflow analysis
- Build time regression detection

**Keywords:** build, deploy, ci/cd, pipeline, github, workflow, test

**Tools Used:**
- GitHub API (via GitHubService)
- CloudWatch Logs for deployment logs
- Custom build metrics

**Sample Output:**
```
📉 BUILD REGRESSION: 23% Slower Builds

Average build time increased from 4m 12s to 5m 10s over last week.

LIKELY CAUSE: New dependency added in commit abc123f
RECOMMENDATION: Review package.json changes, consider caching
```

#### 5. **Database Agent**

**Domain:** Database performance, query optimization, RDS monitoring

**Capabilities:**
- Slow query detection
- Index recommendations
- Connection pool analysis
- Replication lag monitoring
- Storage growth prediction
- Read/write performance analysis

**Keywords:** database, query, index, postgres, mysql, rds, slow query

**Tools Used:**
- `rds:DescribeDBInstances`
- `cloudwatch:GetMetricData` (database metrics)
- `logs:FilterLogEvents` (slow query logs)

**Sample Output:**
```
🗄️ QUERY OPTIMIZATION OPPORTUNITY

Detected 247 slow queries (>500ms) in last 24 hours

TOP OFFENDER:
SELECT * FROM orders WHERE user_id = ?
- Missing index on user_id
- Average execution: 1.2s

RECOMMENDATION:
CREATE INDEX idx_orders_user_id ON orders(user_id);
Expected improvement: 95% faster queries
```

#### 6. **Infrastructure Agent**

**Domain:** Configuration management, drift detection, security posture

**Capabilities:**
- Infrastructure drift detection
- Security group analysis
- IAM policy review
- Configuration compliance checking
- Resource tagging audit
- Network topology analysis

**Keywords:** infrastructure, config, drift, security, iam, vpc, network

**Tools Used:**
- `ec2:DescribeSecurityGroups`
- `iam:ListPolicies`
- `cloudformation:DescribeStacks`
- `config:DescribeConfigurationRecorders`

**Sample Output:**
```
🛡️ SECURITY: Overly Permissive Security Group

sg-abc123 allows 0.0.0.0/0 on port 22 (SSH)

RISK: High - SSH exposed to internet
RECOMMENDATION: Restrict to VPN IP range or remove rule
PRIORITY: Immediate action required
```

### Inter-Agent Communication

**Protocol:**
Agents communicate via standardized `AgentResponse` objects:

```python
@dataclass
class AgentResponse:
    status: TaskStatus  # SUCCESS, PARTIAL, FAILED, PENDING
    data: Dict[str, Any]  # Agent-specific results
    insights: List[Insight]  # Structured findings
    recommendations: List[Recommendation]  # Action items
    agent_type: AgentType  # Source agent identifier
    execution_time: float  # Performance metric
    errors: List[str]  # Any errors encountered
```

**Coordination Example:**

Query: "Overall infrastructure health check"

```
Planner Agent
    ├─> Observability Agent (metrics health)
    ├─> Cost Agent (spending check)
    ├─> Infrastructure Agent (security audit)
    └─> Database Agent (DB performance)
          │
          ├─ Agent 1 returns: 3 insights
          ├─ Agent 2 returns: 2 insights
          ├─ Agent 3 returns: 1 insight
          └─ Agent 4 returns: 2 insights
                │
                └─> Planner synthesizes → 8 total insights ranked by severity
```

### Agent Selection Logic

**Keyword Matching:**
Each agent has predefined keywords. Planner matches user query against these.

**Capability Matching:**
Planner evaluates agent capabilities against task requirements.

**Multi-Agent Tasks:**
Complex queries may invoke multiple agents in sequence or parallel.

**Fallback:**
If no agent matches, Planner handles directly or returns clarification request.

### Health Monitoring

Each agent reports health status:
- **Green:** Operational, responding within SLA
- **Yellow:** Degraded performance, high latency
- **Red:** Unavailable, errors encountered

Health checks performed every 5 minutes (configurable).

---

## 🔄 CI/CD Workflow Monitoring

### Overview

AutoPilot AI integrates with **GitHub Actions** to provide intelligent CI/CD pipeline monitoring, build trend analysis, and failure diagnostics.

### GitHub Integration Architecture

```
┌─────────────────────┐
│  GitHub Repository  │
│                     │
│  • Workflow Runs    │
│  • Commit History   │
│  • Build Logs       │
└──────────┬──────────┘
           │ GitHub API
           ↓
┌─────────────────────┐
│   GitHubService     │
│                     │
│  • Fetch runs       │
│  • Analyze trends   │
│  • Detect failures  │
└──────────┬──────────┘
           │
           ↓
┌─────────────────────┐
│    CI/CD Agent      │
│                     │
│  • Pattern analysis │
│  • Recommendations  │
└─────────────────────┘
```

### Features

#### 1. **Build Monitoring**

**Recent Builds View:**
- Last 10 workflow runs
- Build status (success, failure, cancelled)
- Build duration
- Commit SHA and message
- Triggered by (user/event)
- Branch name
- Timestamp

**API Endpoint:**
```
GET /api/github/builds/recent?limit=10
```

**Response:**
```json
{
  "builds": [
    {
      "id": 987654321,
      "name": "CI Pipeline",
      "status": "completed",
      "conclusion": "success",
      "created_at": "2026-03-09T10:23:45Z",
      "run_time_seconds": 252,
      "commit_sha": "abc123f",
      "commit_message": "Fix user authentication bug",
      "branch": "main",
      "actor": "developer123"
    }
  ]
}
```

#### 2. **Build Trend Analysis**

**Metrics Tracked:**
- Average build time over period
- Success rate percentage
- Failure rate trend
- Build frequency (builds per day)
- Peak build times

**API Endpoint:**
```
GET /api/github/builds/trends?days=7
```

**Analysis:**
```json
{
  "period_days": 7,
  "total_builds": 42,
  "success_rate": 0.88,
  "failure_rate": 0.12,
  "avg_build_time_seconds": 287,
  "trend": "degrading",  // improving/stable/degrading
  "slowest_build": {
    "run_id": 123,
    "duration": 420,
    "date": "2026-03-08"
  }
}
```

#### 3. **Failed Build Tracking**

**Failure Analysis:**
- Recent failed builds
- Failure reasons (extracted from logs)
- Failure frequency by test/stage
- Flaky test detection

**API Endpoint:**
```
GET /api/github/builds/failed?limit=5
```

**Response:**
```json
{
  "failed_builds": [
    {
      "id": 987654320,
      "name": "CI Pipeline",
      "conclusion": "failure",
      "created_at": "2026-03-08T15:32:10Z",
      "commit_sha": "def456g",
      "commit_message": "Add new payment processor",
      "failure_reason": "Test suite failed: test_payment_validation",
      "failed_step": "Run Tests"
    }
  ]
}
```

#### 4. **Build Health Summary**

**Overall Health Metrics:**
- Build health score (0-100)
- Trend direction
- Active issues count
- Recommendations

**API Endpoint:**
```
GET /api/github/builds/health
```

**Response:**
```json
{
  "health_score": 72,
  "status": "fair",  // excellent/good/fair/poor
  "issues": [
    {
      "type": "build_time_regression",
      "severity": "medium",
      "description": "Builds 23% slower than baseline"
    },
    {
      "type": "flaky_tests",
      "severity": "low",
      "description": "test_api_timeout fails intermittently"
    }
  ],
  "recommendations": [
    "Review recent dependency changes",
    "Enable test caching",
    "Investigate flaky test: test_api_timeout"
  ]
}
```

### CI/CD Agent Intelligence

#### Build Time Regression Detection

**Algorithm:**
1. Calculate baseline (average of last 30 successful builds)
2. Compare recent builds (last 7 days) to baseline
3. Flag regression if >15% slower
4. Identify potential causes:
   - New dependencies added
   - Test suite expansion
   - Infrastructure changes

**Output Example:**
```
📉 BUILD REGRESSION DETECTED

Current: 5m 10s | Baseline: 4m 12s | Degradation: 23%

ANALYSIS:
Regression started after commit abc123f (March 6)

LIKELY CAUSES:
1. package.json: Added @aws-sdk/client-s3 (large dependency)
2. Test suite expanded by 18 new tests
3. Docker build cache may be invalidated

RECOMMENDATIONS:
1. Review dependency tree for bundle size
2. Enable GitHub Actions cache for node_modules
3. Split test suite into parallel jobs
4. Consider using smaller base Docker image

ESTIMATED IMPROVEMENT: 40% faster builds
```

#### Test Failure Pattern Recognition

Identifies:
- **Flaky tests:** Pass/fail inconsistently
- **Consistent failures:** Always fail
- **Environmental issues:** Fail in specific conditions
- **Timeout-based failures:** Network-dependent tests

#### Deployment Tracking

Monitors:
- Deployment frequency
- Rollback rate
- Time to production
- Production incident correlation

### Repository Health

**Metrics:**
- Commit frequency
- Contributors count
- Open PR age
- Issue resolution time
- Code review speed

**API Endpoint:**
```
GET /api/github/repo/info
```

**Response:**
```json
{
  "name": "autopilot-ai",
  "description": "Multi-Agent AI SRE System",
  "language": "Python",
  "stars": 42,
  "forks": 8,
  "open_issues": 3,
  "last_commit_at": "2026-03-09T09:15:30Z",
  "contributors_count": 4
}
```

### Chatbot Integration

Users can query CI/CD status naturally:

**Example Queries:**
- "Why did the last build fail?"
- "Show me build trends for the past week"
- "Are there any flaky tests?"
- "How can I speed up my CI pipeline?"

**Agent Response:**
CI/CD Agent fetches data, analyzes patterns, and provides actionable recommendations with GitHub links, commit attribution, and fix suggestions.

---

## ⚙️ Autonomous Jobs

### Overview

**Autonomous Jobs** allow users to schedule recurring AI-powered analysis tasks that run automatically in the background. Think of them as "cron jobs for SRE intelligence."

### Concept

Instead of manually querying the chatbot every day:
- Create a job with a natural language prompt
- Set a schedule (hourly/daily/weekly)
- Let AutoPilot AI execute and report automatically

### User Flow

```
User → Create Job → Schedule → Background Execution → Results Stored
                                        ↓
                                 Alert if issues
                                        ↓
                            View results in Action Center
```

### Job Configuration

**Job Parameters:**
- **Name:** Descriptive label (e.g., "Daily Cost Check")
- **Prompt:** Natural language query (e.g., "Analyze AWS costs and flag anomalies")
- **Schedule:** Execution cadence (hourly/daily/weekly)
- **Enabled:** Active/inactive toggle

### Job Creation Flow

#### Frontend (AutonomousJobsPanel Component)

1. User enters job details:
   - Job name: "Nightly Security Audit"
   - Prompt: "Scan security groups for overly permissive rules"
   - Schedule: Daily

2. Clicks "Create Job"

3. POST request to `/api/ops/jobs`

4. Backend creates job record

5. Scheduler registers job

6. Confirmation displayed

#### Backend Processing

```python
# Job created in ops_state_store
job = {
    "id": 1709982450123,
    "name": "Nightly Security Audit",
    "prompt": "Scan security groups for overly permissive rules",
    "schedule": "daily",
    "enabled": True,
    "last_run_at": None,
    "last_status": None,
    "created_at": "2026-03-09T10:30:45Z",
    "updated_at": "2026-03-09T10:30:45Z"
}

# Scheduler registers job
scheduler.register_autonomous_job(job)

# Schedule library sets up recurring execution
schedule.every(1).days.do(execute_job, job).tag("autonomous-job-1709982450123")
```

### Job Execution

#### Scheduled Execution

```
Background Scheduler
    ↓
Time trigger (hourly/daily/weekly)
    ↓
Execute Job Prompt
    ↓
Send to Planner Agent
    ↓
Agent Processing (same as chat)
    ↓
Store Results
    ↓
Create Action Items (if issues found)
    ↓
Send Alerts (if high severity)
```

#### Manual Trigger ("Run Now")

Users can trigger any job immediately via "Run Now" button:

```
User clicks "Run Now"
    ↓
POST /api/ops/jobs/{job_id}/run
    ↓
Immediate execution (bypasses schedule)
    ↓
Results returned synchronously
```

### Job Results Storage

**OpsStateStore (JSON-backed persistence):**

```json
{
  "job_runs": [
    {
      "job_id": 1709982450123,
      "name": "Nightly Security Audit",
      "status": "SUCCESS",
      "summary": "Found 2 security group issues",
      "insights_count": 2,
      "recommendations_count": 3,
      "started_at": "2026-03-09T01:00:00Z",
      "completed_at": "2026-03-09T01:00:34Z",
      "execution_time_ms": 34240
    }
  ]
}
```

**Job Status Updates:**

After each run:
- `last_run_at` updated
- `last_status` set (SUCCESS/FAILED)
- Job run record created

### Job Management

**Available Actions:**
1. **Run Now:** Immediate execution
2. **Enable/Disable Toggle:** Pause/resume job
3. **Edit:** Modify prompt or schedule
4. **Delete:** Remove job entirely

**UI Features:**
- Visual indicators for enabled/disabled state
- Last run timestamp
- Status badge (success/failed)
- Execution history

### Integration with Action Center

When autonomous jobs find issues:
1. Insights generated
2. Recommendations created
3. **Automatic action items created** in Action Center
4. User sees actionable tasks without manual intervention

**Example:**

Job: "Daily Cost Optimization Check"
Finding: RDS instance over-provisioned
Action Created: "Downgrade RDS from r5.2xlarge to r5.xlarge"
Status: "Suggested"

User can then approve → execute → verify.

### Notification Integration

High-severity findings trigger:
- WebSocket alerts to frontend
- Email notifications (if configured)
- Slack/Teams webhooks (if configured)

### Sample Use Cases

| Job Name | Prompt | Schedule | Purpose |
|----------|--------|----------|---------|
| Daily Cost Check | "Find cost anomalies and over-provisioned resources" | Daily | Cost optimization |
| Security Audit | "Check for security misconfigurations" | Daily | Security posture |
| Build Health Monitor | "Analyze CI/CD trends and detect regressions" | Hourly | DevOps health |
| Database Performance | "Check slow queries and recommend indexes" | Daily | DB optimization |
| Predictive Scaling | "Predict resource saturation in next 7 days" | Weekly | Capacity planning |

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ops/jobs` | GET | List all jobs |
| `/api/ops/jobs` | POST | Create new job |
| `/api/ops/jobs/{id}` | PATCH | Update job (schedule, enable/disable) |
| `/api/ops/jobs/{id}` | DELETE | Delete job |
| `/api/ops/jobs/{id}/run` | POST | Trigger immediate execution |
| `/api/ops/jobs/{id}/runs` | GET | Get execution history |

### Scheduler Architecture

**SchedulerService:**

```python
class SchedulerService:
    def __init__(self, api: AutoPilotAPI):
        self.api = api
        self.running = False
        self.worker_thread = None
        
    def start(self):
        """Start background scheduler thread"""
        self.running = True
        self.worker_thread = threading.Thread(
            target=self._run_loop,
            daemon=True
        )
        self.worker_thread.start()
    
    def _run_loop(self):
        """Main scheduler loop"""
        while self.running:
            schedule.run_pending()
            time.sleep(10)
    
    def register_autonomous_job(self, job):
        """Register job for periodic execution"""
        cadence = job['schedule']
        if cadence == 'hourly':
            schedule.every(1).hours.do(self.execute_job, job)
        elif cadence == 'daily':
            schedule.every(1).days.do(self.execute_job, job)
        elif cadence == 'weekly':
            schedule.every(1).weeks.do(self.execute_job, job)
    
    def execute_job(self, job):
        """Execute job prompt through agent pipeline"""
        response = self.api.query(
            prompt=job['prompt'],
            context={'source': 'autonomous_job'}
        )
        # Store results...
        # Create actions...
        # Send notifications...
```

**Thread Safety:**
- Background thread runs scheduler loop
- Jobs executed in main API context
- State updates protected by locks

### Benefits

**For DevOps Teams:**
- Continuous monitoring without manual effort
- Proactive issue detection
- Automated compliance checking
- Trend analysis over time

**For Businesses:**
- 24/7 SRE vigilance
- Early warning system
- Cost optimization automation
- Reduced manual overhead

---

## 📋 Action Center

### Overview

**Action Center** is a kanban-style workflow board that tracks recommended actions from "Suggested" through "Approved" to "Executed" and finally "Verified." It provides a structured way to manage SRE recommendations.

### Concept

AutoPilot AI generates recommendations (e.g., "Scale ECS service," "Add database index"), but executing them requires human oversight. Action Center bridges the gap between AI recommendations and operational execution.

### Workflow Stages

```
Suggested → Approved → Executed → Verified
```

**Stage Definitions:**

| Stage | Description | Who/What |
|-------|-------------|----------|
| **Suggested** | AI recommends action | AutoPilot AI |
| **Approved** | Human reviews and approves | SRE/DevOps Team |
| **Executed** | Action implemented in infrastructure | Team or automation |
| **Verified** | Outcome validated, issue resolved | Team or monitoring |

### User Flow

#### 1. **Action Creation**

**Source 1: Manual Creation**
- User types action in "Create Action" box
- Example: "Optimize Lambda cold starts"
- Clicks "Add"
- Action appears in "Suggested" column

**Source 2: Autonomous Jobs**
- Background job runs (e.g., "Daily Cost Check")
- Finds issue (e.g., over-provisioned RDS)
- Automatically creates action
- Action appears in "Suggested" column

**Source 3: Chat Recommendations**
- User asks chatbot: "Optimize my database"
- Agent returns recommendations
- User clicks "Create Action" on recommendation
- Action added to Action Center

#### 2. **Action Progression**

Each action card has a dropdown to change status:

```
┌─────────────────────────────────┐
│  Reduce RDS from r5.2xl to r5.xl│
│                                 │
│  Status: [Suggested ▼]          │
│          └─ Suggested           │
│             Approved            │
│             Executed            │
│             Verified            │
└─────────────────────────────────┘
```

User selects new status → Action moves to corresponding column.

#### 3. **Collaboration**

Team members can:
- Review suggested actions
- Discuss in external tools (Slack, Teams)
- Approve critical actions
- Mark as executed after implementation
- Verify outcomes

### UI Layout

**Four-Column Kanban Board:**

```
┌──────────────┬──────────────┬──────────────┬──────────────┐
│  Suggested   │   Approved   │   Executed   │   Verified   │
│      📋      │      ✅      │      ▶       │      ✓       │
├──────────────┼──────────────┼──────────────┼──────────────┤
│ Action 1     │ Action 3     │ Action 5     │ Action 7     │
│ Action 2     │ Action 4     │ Action 6     │              │
└──────────────┴──────────────┴──────────────┴──────────────┘
```

**Action Card Contents:**
- Title (brief description)
- Timestamp (created_at, updated_at)
- Status dropdown
- Optional: tags, priority, assignee (future enhancement)

### Data Model

**Action Structure:**

```json
{
  "id": 1709982450123,
  "title": "Scale ECS service from 2 to 4 tasks",
  "status": "Approved",
  "created_at": "2026-03-09T10:30:00Z",
  "updated_at": "2026-03-09T11:15:23Z",
  "metadata": {
    "source": "autonomous_job",
    "job_id": 12345,
    "severity": "high",
    "cost_impact_inr": "16308"
  }
}
```

**Storage:**
- Persisted in `local_kb/configs/ops_state.json`
- Survives server restarts
- Thread-safe with locking

### Component Architecture

**ActionCenter Component (React):**

```jsx
function ActionCenter({ actions, onCreateAction, onUpdateAction }) {
  // Group actions by status
  const grouped = useMemo(() => {
    return STAGES.reduce((acc, stage) => {
      acc[stage] = actions.filter(item => item.status === stage)
      return acc
    }, {})
  }, [actions])
  
  // Render 4 columns
  return (
    <div className="grid grid-cols-4">
      {STAGES.map(stage => (
        <Column
          key={stage}
          title={stage}
          actions={grouped[stage]}
          onUpdateAction={onUpdateAction}
        />
      ))}
    </div>
  )
}
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/ops/actions` | GET | List all actions |
| `/api/ops/actions` | POST | Create new action |
| `/api/ops/actions/{id}` | PATCH | Update action status |
| `/api/ops/actions/{id}` | DELETE | Delete action |

**Create Action:**
```
POST /api/ops/actions
Authorization: Bearer <token>

Request:
{
  "title": "Add index to user_id column in orders table"
}

Response:
{
  "id": 1709982450123,
  "title": "Add index to user_id column in orders table",
  "status": "Suggested",
  "created_at": "2026-03-09T12:00:00Z",
  "updated_at": "2026-03-09T12:00:00Z"
}
```

**Update Action Status:**
```
PATCH /api/ops/actions/1709982450123
Authorization: Bearer <token>

Request:
{
  "status": "Approved"
}

Response:
{
  "id": 1709982450123,
  "title": "Add index to user_id column in orders table",
  "status": "Approved",
  "created_at": "2026-03-09T12:00:00Z",
  "updated_at": "2026-03-09T12:05:30Z"
}
```

### Integration with Other Features

**From Autonomous Jobs:**
```python
# Job finds issue
if severity >= Severity.HIGH:
    action = ops_state_store.create_action(
        title=recommendation['title'],
        metadata={'source': 'autonomous_job', 'job_id': job['id']}
    )
```

**From Chat Recommendations:**
```javascript
// User clicks "Add to Action Center" button
const createAction = async (recommendation) => {
  await axios.post('/api/ops/actions', {
    title: recommendation.title
  })
  // Refresh action center
}
```

### Workflow Automation (Future Enhancement)

**Potential Integrations:**
- **Terraform:** Generate terraform plan on approval
- **GitHub Issues:** Create issue when action created
- **Slack:** Notify team when action needs approval
- **Jira:** Sync with Jira tickets
- **Execution Scripts:** Auto-execute low-risk approved actions

### Use Case Example

**Scenario:** Autonomous job detects over-provisioned RDS

1. **Detection (Autonomous Job):**
   - Job runs: "Daily Cost Optimization"
   - Finds: RDS r5.2xlarge at 27% utilization
   - Creates recommendation

2. **Action Created:**
   - Title: "Downgrade RDS from r5.2xlarge to r5.xlarge"
   - Status: "Suggested"
   - Metadata: Cost savings ₹20,750/month

3. **Review:**
   - DevOps lead reviews action
   - Checks recommendation details
   - Confirms low risk
   - Changes status to "Approved"

4. **Execution:**
   - Engineer runs AWS CLI command
   - RDS instance modified
   - Status changed to "Executed"

5. **Verification:**
   - Monitoring confirms instance stable
   - Cost Explorer shows reduced spend
   - Status changed to "Verified"

6. **Outcome:**
   - Action archived (or deleted)
   - Savings tracked
   - Process documented

### Benefits

**Visibility:**
- Clear view of pending actions
- Track progress from idea to completion
- Audit trail of changes

**Collaboration:**
- Team members see same board
- Status updates shared in real-time
- Accountability for execution

**Integration:**
- Bridges AI recommendations and human execution
- Reduces recommendation fatigue
- Ensures follow-through

---

## 🔔 Real-Time Monitoring & Alerts

### Overview

AutoPilot AI provides **real-time infrastructure monitoring** with WebSocket-based alerts that notify users instantly when issues arise.

### Architecture

```
Background Scheduler → Detects Issue → Creates Alert → WebSocket Broadcast
                                                              ↓
                                                    Connected Clients
                                                              ↓
                                                    LiveAlertsPanel
```

### Alert Sources

#### 1. **CloudWatch Alarms**

Monitors AWS CloudWatch alarm state changes:
- CPU utilization thresholds exceeded
- Memory pressure
- Disk space warnings
- Application errors

#### 2. **CI/CD Events**

GitHub workflow events:
- Build failures
- Deployment errors
- Test regressions
- Long build times

#### 3. **Cost Anomalies**

AWS billing alerts:
- Spend exceeding budget
- Cost spike detection
- Unexpected service charges

#### 4. **Autonomous Job Findings**

High-severity issues from background jobs:
- Security misconfigurations
- Performance degradation
- Resource saturation predictions

### Alert Structure

```json
{
  "id": "alert-1709982450123",
  "timestamp": "2026-03-09T12:34:56Z",
  "severity": "CRITICAL",
  "title": "ECS Service CPU Critical",
  "description": "api-service averaging 92% CPU over last 15 minutes",
  "source": "CloudWatch",
  "service": "ECS",
  "resource": "api-service",
  "cost_impact_inr": null,
  "recommendation": "Scale to 4 tasks immediately",
  "commit_attribution": null,
  "dismissed": false
}
```

### Alert Severity Levels

| Severity | Color | Use Case | Response Time |
|----------|-------|----------|---------------|
| **CRITICAL** | 🔴 Red | Service down, data loss risk | Immediate |
| **HIGH** | 🟠 Orange | Performance degradation | <1 hour |
| **MEDIUM** | 🟡 Yellow | Potential future issues | <24 hours |
| **LOW** | 🔵 Blue | Informational, optimization | When convenient |

### WebSocket Protocol

#### Connection

```javascript
// Frontend connects
const ws = new WebSocket('ws://localhost:8000/ws/alerts')

ws.onopen = () => {
  console.log('Connected to alerts stream')
}

ws.onmessage = (event) => {
  const alert = JSON.parse(event.data)
  // Display alert in UI
  addAlert(alert)
}
```

#### Message Format

**Server → Client:**
```json
{
  "type": "alert",
  "data": {
    "id": "alert-123",
    "severity": "HIGH",
    "title": "Build failed on main branch",
    ...
  }
}
```

**Client → Server (acknowledgment):**
```json
{
  "type": "dismiss",
  "alert_id": "alert-123"
}
```

### LiveAlertsPanel Component

**Features:**
- Real-time alert stream
- Severity filtering (show only critical, high, etc.)
- Dismiss functionality
- Auto-scroll to latest
- Sound notifications (optional)
- Visual badges for unread count

**UI Elements:**

```
┌─────────────────────────────────────────┐
│  🔔 Live Alerts              [Filter ▼] │
├─────────────────────────────────────────┤
│  🔴 CRITICAL                             │
│  ECS Service CPU Critical               │
│  api-service averaging 92% CPU          │
│  Recommendation: Scale to 4 tasks       │
│  [Dismiss]                   2m ago     │
├─────────────────────────────────────────┤
│  🟡 MEDIUM                               │
│  Build time regression detected         │
│  Builds 18% slower than baseline        │
│  [Dismiss]                   15m ago    │
└─────────────────────────────────────────┘
```

### Alert Persistence

**In-Memory (Session):**
- Alerts stored in frontend state during session
- Lost on page refresh

**Database (Optional):**
- Critical alerts can be persisted to ops_state_store
- Historical alert tracking
- Alert analytics

### Alert Lifecycle

```
Alert Triggered
    ↓
WebSocket Broadcast
    ↓
Displayed in UI
    ↓
User Acknowledges
    ↓
Alert Dismissed (removed from view)
    ↓
(Optional) Logged to history
```

### Integration with Action Center

**High-Severity Alerts → Auto-Create Actions:**

```python
if alert['severity'] in [Severity.CRITICAL, Severity.HIGH]:
    if alert.get('recommendation'):
        ops_state_store.create_action(
            title=alert['recommendation'],
            metadata={
                'alert_id': alert['id'],
                'source': 'real_time_alert'
            }
        )
```

Result: Alert appears in LiveAlertsPanel AND Action Center simultaneously.

### Notification Channels

**Current:**
- WebSocket (frontend)
- In-app notifications

**Future:**
- Email
- Slack webhooks
- Microsoft Teams
- PagerDuty integration
- SMS (Twilio)

### Health Check Integration

**Agent Health Monitoring:**

Every 5 minutes:
1. Check each agent's health
2. If agent degraded/unavailable:
   - Create alert
   - Broadcast to WebSocket
   - Log to ops_state

**Service Health Monitoring:**

Services monitored:
- AWS Bedrock connectivity
- Knowledge Base availability
- CloudWatch API access
- GitHub API rate limits

### Sample Alerts

#### Alert 1: Critical Infrastructure Issue

```
🔴 CRITICAL
ECS Service Unhealthy

Service: api-service
Cluster: production

All tasks failing health checks. Service unresponsive.

Recommendation: Check recent deployments. Rollback if necessary.
Execute: aws ecs update-service --cluster production --service api-service --force-new-deployment

12 seconds ago | [Dismiss]
```

#### Alert 2: Cost Spike

```
🟠 HIGH
AWS Cost Spike Detected

Current spend: ₹52,340 (↑ 35% from yesterday)

Top contributor: RDS - ₹18,200

Recommendation: Review RDS usage. Possible snapshot retention increase.

5 minutes ago | [Dismiss]
```

#### Alert 3: Build Failure

```
🟡 MEDIUM
Build Failed on Main Branch

Commit: abc123f - "Update payment processor"
Author: developer@example.com

Failed step: Run Tests
Error: test_payment_validation failed

View logs: [GitHub Link]

18 minutes ago | [Dismiss]
```

### Alert Deduplication

**Problem:** Same issue triggers multiple alerts

**Solution:**
- Alert fingerprinting (hash of alert key attributes)
- Suppress duplicate alerts within time window (e.g., 5 minutes)
- Increment alert counter instead of creating new alert

### Performance Considerations

**Scalability:**
- WebSocket connection per client
- Alerts broadcast to all connected clients
- Limit alert rate (max 10 alerts/minute per source)

**Throttling:**
- Batch similar alerts
- Rate limit by severity
- Prevent alert storms

---

## 📚 Knowledge Base System

### Overview

AutoPilot AI includes a **modular knowledge base** system for storing and retrieving SRE best practices, runbooks, and organizational knowledge.

### Two Modes

#### 1. **Local Knowledge Base** (Development)

**Use Case:** Local development, testing, no AWS required

**Implementation:**
- File-based storage in `./local_kb/documents/`
- Simple keyword search
- No external dependencies
- Fast setup

**Structure:**
```
local_kb/
├── documents/         # Knowledge articles (markdown)
│   ├── runbook_ecs_scaling.md
│   ├── cost_optimization_rds.md
│   └── incident_response.md
├── metrics/           # Metric definitions
└── configs/           # Configuration files
    ├── ops_state.json
    └── autopilot_auth.db
```

**Search Algorithm:**
- Keyword matching (case-insensitive)
- TF-IDF scoring
- Returns top N relevant documents

**Advantages:**
- Zero configuration
- Works offline
- Fast iteration
- No AWS costs

**Limitations:**
- Basic search (no semantic understanding)
- Manual document management
- No vector embeddings

#### 2. **AWS Bedrock Knowledge Base** (Production)

**Use Case:** Production deployment, advanced semantic search

**Implementation:**
- S3 backend for document storage
- Amazon Bedrock Knowledge Base with vector embeddings
- Semantic search powered by embeddings
- Automatic indexing

**Architecture:**
```
Documents → S3 Bucket → Bedrock KB → Vector Store
                                           ↓
                            User Query → Semantic Search → Relevant Docs
```

**Setup:**
1. Create S3 bucket
2. Upload documents
3. Create Bedrock Knowledge Base
4. Configure data source (S3)
5. Index documents
6. Set knowledge base ID in config

**Advantages:**
- Semantic search (understands meaning, not just keywords)
- Automatic document ingestion
- Scalable
- Managed service

**Configuration:**
```python
# config.py
USE_LOCAL_KB = False  # Use AWS Bedrock KB
BEDROCK_KB_ID = "YOUR_KB_ID_HERE"
BEDROCK_KB_S3_BUCKET = "autopilot-knowledge-base"
```

### Knowledge Base Factory Pattern

**Abstraction Layer:**

```python
class KnowledgeBaseInterface(ABC):
    @abstractmethod
    def query(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search knowledge base"""
        pass
    
    @abstractmethod
    def add_document(self, content: str, metadata: Dict) -> str:
        """Add new document"""
        pass
```

**Implementation Selection:**

```python
def get_knowledge_base() -> KnowledgeBaseInterface:
    if config.USE_LOCAL_KB:
        return LocalKnowledgeBase()
    else:
        return BedrockKnowledgeBase()
```

**Benefit:** Switch between implementations seamlessly.

### Knowledge Base Usage in Agents

**Query Flow:**

```python
# Agent needs best practices
kb = get_knowledge_base()

results = kb.query(
    query="How to optimize RDS performance for PostgreSQL?",
    limit=3
)

# Results fed to LLM as context
prompt = f"""
Based on these best practices:
{results}

Analyze the user's database performance...
"""

response = bedrock_client.invoke(prompt)
```

### Document Structure

**Recommended Format (Markdown):**

```markdown
# Runbook: ECS Service Scaling

## Overview
Guide for scaling ECS services in production.

## When to Scale

- CPU utilization > 70% for 10+ minutes
- Memory utilization > 75%
- Request queue depth increasing

## Scaling Procedure

1. Check current task count:
   aws ecs describe-services --cluster production --services api-service

2. Determine target count (current + 50%)

3. Update service:
   aws ecs update-service --cluster production --service api-service --desired-count 4

4. Monitor task health:
   aws ecs describe-services --cluster production --services api-service | jq '.services[].runningCount'

## Rollback

If issues arise:
aws ecs update-service --cluster production --service api-service --desired-count <original>

## Cost Impact

Each task costs ₹996/month (t3.medium, 24/7)

## Related

- Runbook: RDS Scaling
- Alert: ECS CPU Critical
```

**Metadata:**
- Title
- Category (runbook, best-practice, incident-report)
- Tags (ecs, scaling, performance)
- Last updated
- Author

### Knowledge Base Seeding

**Initial Documents:**

1. **Runbooks:**
   - ECS scaling procedures
   - RDS maintenance windows
   - Lambda cold start optimization
   - Incident response workflows

2. **Best Practices:**
   - Cost optimization strategies
   - Security hardening checklists
   - Database indexing guidelines
   - CI/CD pipeline patterns

3. **Organizational Knowledge:**
   - Team contacts
   - Escalation procedures
   - Post-mortem templates
   - Architecture diagrams

**Seeding Script:**

```python
# scripts/seed_kb.py
kb = get_knowledge_base()

documents = [
    ("runbook_ecs_scaling.md", {"category": "runbook", "tags": ["ecs"]}),
    ("best_practice_rds_optimization.md", {"category": "best-practice", "tags": ["rds"]}),
]

for filename, metadata in documents:
    content = Path(f"./local_kb/documents/{filename}").read_text()
    kb.add_document(content, metadata)
```

### Benefits

**For Agents:**
- Access to organizational knowledge
- Consistent recommendations
- Reduced hallucination (grounded in facts)

**For Teams:**
- Centralized knowledge repository
- Easy knowledge sharing
- Onboarding resource

**For Operations:**
- Faster incident resolution
- Standardized procedures
- Historical context

---

## ⚡ Dynamic AWS SDK Execution

### Overview

The **defining feature** of AutoPilot AI: instead of hardcoded AWS API calls, Claude Sonnet dynamically decides which AWS services to query, which operations to perform, and what parameters to use - all via natural language understanding.

### The Problem with Static APIs

**Traditional Approach:**
```python
# Hardcoded - developer must know exact parameters
cloudwatch = boto3.client('cloudwatch')
response = cloudwatch.list_metrics(
    Namespace='AWS/EC2',
    MetricName='CPUUtilization',
    MaxRecords=20  # ❌ ERROR - not a valid parameter!
)
```

**Problems:**
- Developers must memorize API signatures
- Brittle code breaks when APIs change
- Can't adapt to new AWS services automatically
- Parameter validation errors

### AutoPilot AI's Solution

**Dynamic Tool Execution:**

```
User Query: "Show me EC2 instances with high CPU"
            ↓
    Claude Sonnet Analyzes
            ↓
    Decides: Need to call CloudWatch
            ↓
    Tool Request: {
        "tool": "aws_api_executor",
        "service": "cloudwatch",
        "operation": "list_metrics",
        "parameters": {
            "Namespace": "AWS/EC2",
            "MetricName": "CPUUtilization"
        }
    }
            ↓
    AWSAPIExecutor.execute()
            ↓
    Boto3 Dynamic Client
            ↓
    AWS API
            ↓
    Results → Claude → Natural Language Answer
```

### How It Works

#### Component: AWSAPIExecutor

```python
class AWSAPIExecutor:
    """Execute arbitrary AWS SDK operations dynamically."""
    
    def execute(
        self,
        service: str,
        operation: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Args:
            service: AWS service name (e.g., 'cloudwatch', 'ec2', 'rds')
            operation: Operation name (e.g., 'list_metrics', 'describe_instances')
            parameters: Operation parameters as dict
        
        Returns:
            API response
        """
        # Create boto3 client for specified service
        client = self.session.client(service, region_name=self.region)
        
        # Get operation method
        operation_method = getattr(client, operation)
        
        # Execute with provided parameters
        response = operation_method(**parameters)
        
        return response
```

**Key Features:**
- **Service-agnostic:** Works with any AWS service boto3 supports
- **Operation-agnostic:** Works with any operation
- **Self-correcting:** If parameters wrong, error returned to LLM, LLM retries

#### Tool Definition for Bedrock

```python
def get_tool_definition() -> Dict[str, Any]:
    return {
        "name": "aws_api_executor",
        "description": "Execute AWS SDK operations dynamically",
        "input_schema": {
            "type": "object",
            "properties": {
                "service": {
                    "type": "string",
                    "description": "AWS service name (e.g., ec2, rds, cloudwatch)"
                },
                "operation": {
                    "type": "string",
                    "description": "Operation name (e.g., describe_instances)"
                },
                "parameters": {
                    "type": "object",
                    "description": "Operation parameters as key-value pairs"
                }
            },
            "required": ["service", "operation", "parameters"]
        }
    }
```

### Example: User Query Processing

**Query:** "Show me all RDS databases and their CPU usage"

**Claude's Thinking (internal):**
1. User wants RDS database info
2. Need to list RDS instances
3. Then get CloudWatch metrics for each
4. Two API calls required

**Tool Calls Made by Claude:**

**Call 1: List RDS Instances**
```json
{
  "tool": "aws_api_executor",
  "service": "rds",
  "operation": "describe_db_instances",
  "parameters": {}
}
```

**Response:**
```json
{
  "DBInstances": [
    {
      "DBInstanceIdentifier": "production-db",
      "DBInstanceClass": "db.r5.xlarge",
      "Engine": "postgres",
      ...
    }
  ]
}
```

**Call 2: Get CloudWatch Metrics**
```json
{
  "tool": "aws_api_executor",
  "service": "cloudwatch",
  "operation": "get_metric_statistics",
  "parameters": {
    "Namespace": "AWS/RDS",
    "MetricName": "CPUUtilization",
    "Dimensions": [
      {"Name": "DBInstanceIdentifier", "Value": "production-db"}
    ],
    "StartTime": "2026-03-09T00:00:00Z",
    "EndTime": "2026-03-09T12:00:00Z",
    "Period": 3600,
    "Statistics": ["Average"]
  }
}
```

**Response:**
```json
{
  "Datapoints": [
    {"Average": 23.5, "Timestamp": "2026-03-09T10:00:00Z"},
    {"Average": 25.1, "Timestamp": "2026-03-09T11:00:00Z"}
  ]
}
```

**Claude's Final Answer:**
```
You have 1 RDS database:

Production DB (production-db)
- Type: PostgreSQL on db.r5.xlarge
- CPU Usage: 24% average over last 12 hours
- Status: Healthy

💡 Cost Optimization Opportunity:
CPU utilization is low (24%). Consider downsizing to db.r5.large
for ₹10,375/month savings (50% reduction).
```

### Self-Correction Example

**Scenario:** Claude makes a mistake with parameters

**Call 1: Incorrect**
```json
{
  "tool": "aws_api_executor",
  "service": "cloudwatch",
  "operation": "list_metrics",
  "parameters": {
    "Namespace": "AWS/EC2",
    "MaxRecords": 20  // ❌ Not a valid parameter
  }
}
```

**Error Response:**
```json
{
  "error": "Unknown parameter: MaxRecords",
  "valid_parameters": ["Namespace", "MetricName", "Dimensions", "NextToken"]
}
```

**Call 2: Corrected**
```json
{
  "tool": "aws_api_executor",
  "service": "cloudwatch",
  "operation": "list_metrics",
  "parameters": {
    "Namespace": "AWS/EC2"
  }
}
```

**Success:** Claude learns from error and retries correctly.

### Supported AWS Services

**Currently Supported (via boto3):**
- CloudWatch (metrics, alarms, logs)
- EC2 (instances, volumes, snapshots)
- RDS (databases, clusters)
- ECS (clusters, services, tasks)
- Lambda (functions, invocations)
- S3 (buckets, objects)
- Cost Explorer (costs, forecasts)
- ELB (load balancers)
- IAM (users, roles, policies)
- CloudFormation (stacks)
- SNS (topics, subscriptions)
- And any other boto3-supported service!

**Extensibility:**
No code changes needed to support new AWS services. As soon as boto3 adds a service, Claude can use it.

### Integration with Agents

**Observability Agent Example:**

```python
def analyze_performance(self, user_query: str):
    """Analyze infrastructure performance dynamically."""
    
    system_prompt = """
    You are an AWS observability expert. Use the aws_api_executor tool
    to query AWS services and analyze performance.
    
    Available services: cloudwatch, ec2, ecs, rds, lambda
    
    Analyze the user's question and determine which AWS APIs to call.
    """
    
    response = bedrock_client.invoke_with_tools(
        system_prompt=system_prompt,
        user_prompt=user_query,
        tools=[aws_executor.get_tool_definition()],
        tool_executor=self.execute_tool,
        max_iterations=10
    )
    
    return response

def execute_tool(self, tool_name, tool_input):
    """Handle tool execution requests from Claude."""
    if tool_name == "aws_api_executor":
        return aws_executor.execute(**tool_input)
```

**Result:** Agent can answer ANY performance question by dynamically querying AWS.

### Advantages

#### 1. **Adaptive Intelligence**
- No predefined query templates
- Handles novel questions
- Learns from AWS API responses

#### 2. **Future-Proof**
- New AWS services automatically supported
- No code updates for API changes
- Scales with AWS ecosystem

#### 3. **Error Recovery**
- Self-correcting on parameter errors
- Retries with valid parameters
- Learns from mistakes

#### 4. **Developer Experience**
- No AWS API memorization required
- Natural language → AWS calls
- Reduces development time by 80%

#### 5. **Comprehensive Coverage**
- Supports ALL AWS services
- Any operation, any parameter
- No artificial limitations

### Security & Access Control

**Credentials:**
- User's AWS credentials used for all API calls
- Decrypted on-demand, never logged
- Scoped to user's IAM permissions

**Rate Limiting:**
- Respects AWS API rate limits
- Implements exponential backoff
- Prevents API throttling

**Auditing:**
- All API calls logged
- User attribution
- Timestamp and parameters recorded

### Comparison: Before vs After

**Before (Hardcoded):**
```python
# Developer writes:
def check_ec2_cpu():
    aw = boto3.client('cloudwatch')
    response = cw.get_metric_statistics(
        Namespace='AWS/EC2',
        MetricName='CPUUtilization',
        Dimensions=[{'Name': 'InstanceId', 'Value': 'i-12345'}],
        StartTime=datetime.utcnow() - timedelta(hours=1),
        EndTime=datetime.utcnow(),
        Period=300,
        Statistics=['Average']
    )
    # ... manual processing
```

**After (Dynamic):**
```python
# User asks:
"Show EC2 CPU usage for the last hour"

# Claude autonomously:
# 1. Decides to use CloudWatch
# 2. Determines correct operation
# 3. Figures out parameters
# 4. Executes API call
# 5. Analyzes results
# 6. Returns natural language answer

# Developer writes: ZERO code!
```

**Lines of Code:**
- Before: ~50 lines per query type
- After: 0 lines (LLM handles it)

**Maintainability:**
- Before: Update code for every new query
- After: No code changes, just ask

---

## 🛠️ Technical Stack

### Frontend

**Core:**
- **React 18:** Modern component architecture
- **Vite:** Lightning-fast build tool
- **Tailwind CSS:** Utility-first styling

**UI Components:**
- Custom component library (shadcn/ui inspired)
- Lucide React icons
- Responsive design (mobile, tablet, desktop)

**State Management:**
- React useState/useEffect hooks
- Component local state
- WebSocket state synchronization

**HTTP Client:**
- Axios for REST API calls
- Bearer token authentication

**Real-Time:**
- Native WebSocket API
- Auto-reconnect logic

### Backend

**Framework:**
- **FastAPI:** Modern async Python web framework
- **Uvicorn:** ASGI server
- **Pydantic:** Data validation

**AI/ML:**
- **AWS Bedrock:** Claude Sonnet 3.5 LLM
- **Anthropic Messages API:** Tool use (function calling)

**AWS Integration:**
- **Boto3:** AWS SDK for Python
- Dynamic client creation
- Session management

**Database:**
- **SQLite:** User authentication, account connections
- **JSON Files:** Operational state (actions, jobs)

**Background Processing:**
- **Python threading:** Background scheduler
- **Schedule library:** Cron-like job scheduling

**Security:**
- **Cryptography (Fernet):** Symmetric encryption for credentials
- **PBKDF2:** Password hashing
- **Secrets:** Cryptographically secure random tokens

### External Services

**AWS Services:**
- **Bedrock:** LLM inference
- **Bedrock Knowledge Base:** Semantic search (optional)
- **S3:** Document storage (optional)
- **CloudWatch:** Metrics, logs, alarms
- **Cost Explorer:** Billing data
- **EC2, RDS, ECS, Lambda, etc.:** Infrastructure monitoring

**GitHub:**
- **GitHub API:** Repository data, workflow runs
- **GitHub Actions:** CI/CD monitoring
- **OAuth:** User authentication (optional)

### Development Tools

**Python:**
- **Python 3.9+:** Language runtime
- **Virtual Environment:** Dependency isolation
- **pip:** Package management

**Node.js:**
- **Node 18+:** Frontend tooling
- **npm:** Package management

**Deployment:**
- **PowerShell Script:** One-command startup
- **Environment Variables:** Configuration management

### Architecture Patterns

**Multi-Agent System:**
- Planner-coordinator pattern
- Specialized domain agents
- Inter-agent communication protocol

**Factory Pattern:**
- Knowledge base factory
- Client abstraction

**Repository Pattern:**
- AuthStore for user data
- OpsStateStore for operational data

**Observer Pattern:**
- WebSocket real-time updates
- Event-driven alerts

**Strategy Pattern:**
- Pluggable agents
- Dynamic tool execution

### File Structure

```
AWS Project/
├── frontend/                    # React frontend
│   ├── src/
│   │   ├── components/          # UI components
│   │   ├── lib/                 # Utilities
│   │   ├── App.jsx              # Main app
│   │   └── main.jsx             # Entry point
│   ├── package.json
│   └── vite.config.js
├── agents/                      # Specialized agents
│   ├── planner_agent.py
│   ├── observability_agent.py
│   ├── cost_agent.py
│   ├── cicd_agent.py
│   ├── db_agent.py
│   └── infra_agent.py
├── services/                    # Core services
│   ├── bedrock_client.py
│   ├── aws_api_executor.py
│   ├── github_service.py
│   ├── scheduler.py
│   ├── auth_store.py
│   ├── ops_state_store.py
│   └── knowledge_base*.py
├── models/                      # Data models
│   ├── core_models.py
│   └── agent_protocol.py
├── api/                         # API layer
│   └── routes.py
├── app.py                       # FastAPI application
├── config.py                    # Configuration
├── requirements.txt             # Python dependencies
└── run.ps1                      # Startup script
```

### Configuration

**Environment Variables (.env):**
```env
# AWS Configuration
AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your_key_here
AWS_SECRET_ACCESS_KEY=your_secret_here

# Bedrock Configuration
BEDROCK_MODEL_PLANNER=anthropic.claude-3-5-sonnet-20241022-v2:0
BEDROCK_MODEL_OBSERVABILITY=anthropic.claude-3-5-sonnet-20241022-v2:0
# ... other agent models

# GitHub Integration
GITHUB_TOKEN=your_github_token
GITHUB_REPO_OWNER=your_username
GITHUB_REPO_NAME=your_repo

# Knowledge Base
USE_LOCAL_KB=true
BEDROCK_KB_ID=your_kb_id  # if using AWS KB

# Server Configuration
API_PORT=8000
ENABLE_CORS=true
CORS_ORIGINS=http://localhost:5173

# Scheduler
HEALTH_CHECK_INTERVAL_MINUTES=5
```

### Performance Characteristics

**Response Times:**
- Authentication: <100ms
- Chat query (simple): 2-4 seconds
- Chat query (complex, multi-agent): 5-10 seconds
- Autonomous job: 10-30 seconds (depending on complexity)
- WebSocket alert latency: <500ms

**Scalability:**
- Concurrent users: 50+ (limited by AWS API rate limits)
- WebSocket connections: 100+ per server
- Background jobs: Unlimited (scheduled)

**Resource Usage:**
- Backend memory: ~200MB baseline
- Frontend bundle: ~500KB gzipped
- Database size: <10MB (typical)

---

## 🎯 Key Differentiators

### 1. **India-First Design**
- All costs in ₹ (Indian Rupees)
- AWS India region optimization
- ROI calculations with local pricing

### 2. **Dynamic AWS SDK Execution**
- No hardcoded API calls
- LLM decides which APIs to call
- Self-correcting on errors
- Future-proof for new AWS services

### 3. **Multi-Agent Intelligence**
- Specialized agents for different domains
- Coordinated workflows
- Better than single-agent systems

### 4. **Semantic Insights, Not Just Metrics**
- Business context, not raw numbers
- Root cause analysis
- Actionable recommendations
- ROI calculations

### 5. **Autonomous Operations**
- Background jobs run without human intervention
- Proactive issue detection
- Automated action creation

### 6. **End-to-End Workflow**
- From detection → recommendation → action → verification
- Action Center tracks full lifecycle

### 7. **GitHub Integration**
- CI/CD monitoring
- Build trend analysis
- Commit attribution
- Unique in SRE tools

### 8. **Modular Architecture**
- Local development mode (no AWS needed)
- Production-ready AWS mode
- Easy switching

---

## 📊 Demo Scenarios

### Scenario 1: Worker Pool Optimization

**User queries:** "Analyze my ECS services for performance issues"

**System response:**
```
🔴 CRITICAL: Worker Pool Undersized

ROOT CAUSE:
Your Celery worker pool (2 tasks) can't keep up with Redis queue 
depth (1,247 jobs). Average job processing is delayed by 12 seconds.

BUSINESS IMPACT:
- User requests timing out
- Background jobs backing up
- Potential data processing delays

RECOMMENDATION:
Scale ECS service from 2 to 4 tasks

COST ANALYSIS:
├─ Current Cost: ₹1,992/month
├─ Proposed Cost: ₹3,984/month
├─ Additional Cost: +₹1,992/month
└─ Savings: ₹18,300/month (prevents Redis over-provisioning)

NET SAVINGS: ₹16,308/month (818% ROI)
RISK: Low
TIME: 5 minutes

MIGRATION SCRIPT:
aws ecs update-service \
  --cluster production \
  --service api-service \
  --desired-count 4
```

### Scenario 2: Cost Anomaly Detection

**User queries:** "Why did my AWS costs increase?"

**System response:**
```
⚠️ COST ANOMALY: RDS Over-Provisioned

DETECTION:
RDS costs increased by ₹11,620 (40%) compared to baseline

ROOT CAUSE:
Database instance db.r5.2xlarge is severely underutilized:
- CPU: 23% (target: 60-70%)
- Memory: 31% (target: 60-70%)

RECOMMENDATION:
Downgrade from db.r5.2xlarge to db.r5.xlarge

SAVINGS: ₹20,750/month (50% reduction)
ANNUAL IMPACT: ₹2.49L/year

DOWNTIME: 5-10 minutes
PERFORMANCE IMPACT: None (can handle 54% utilization)

IMPLEMENTATION:
aws rds modify-db-instance \
  --db-instance-identifier production-db \
  --db-instance-class db.r5.xlarge \
  --apply-immediately
```

### Scenario 3: Build Regression Detection

**Autonomous job runs:** "Analyze CI/CD trends"

**System generates:**
```
📉 BUILD REGRESSION: 23% Slower Builds

Average build time increased from 4m 12s to 5m 10s

CAUSE:
New dependency @aws-sdk/client-s3 added in commit abc123f

RECOMMENDATIONS:
1. Review dependency tree
2. Enable GitHub Actions cache
3. Split tests into parallel jobs

ESTIMATED IMPROVEMENT: 40% faster builds

ACTION CREATED: "Optimize CI pipeline"
```

---

## 🏆 Conclusion

**AutoPilot AI** represents a paradigm shift in SRE tooling:

✅ **Intelligent:** AI-powered analysis, not just dashboards  
✅ **Adaptive:** Dynamic AWS SDK execution  
✅ **Proactive:** Autonomous jobs detect issues before they escalate  
✅ **Actionable:** From insight to execution with Action Center  
✅ **India-Focused:** Cost optimization in ₹ for Indian startups  
✅ **Comprehensive:** End-to-end SRE workflow automation  

### For Hackathon Judges

This system demonstrates:

1. **AWS Bedrock Integration:** Claude Sonnet for intelligent analysis
2. **Multi-Agent Architecture:** Coordinated specialist agents
3. **Real-World Application:** Solves real SRE pain points
4. **Production-Ready:** Authentication, persistence, monitoring
5. **Innovation:** Dynamic tool calling, not static rules
6. **User Experience:** Clean UI, natural language interface
7. **Scalability:** Modular design, extensible architecture

---

**Team Ragnar** | **March 2026**

*Built for the AWS AI for Bharat Hackathon*
