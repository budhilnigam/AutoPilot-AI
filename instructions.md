# 🚀 AutoPilotAI — Copilot Instruction File v2 (Strict Architecture Lock)

---

## 🔷 Project Name

**AutoPilotAI — Multi-Agent AI SRE System**

---

## 🔷 Core Mission

AutoPilotAI is a **production-grade, multi-agent AI SRE system** built on Amazon Bedrock for Indian startups.

It transforms raw DevOps telemetry into:

* Semantic insights
* Root cause analysis
* Cost-aware optimization plans (in INR)
* Predictive infrastructure alerts
* CI/CD regression detection
* Database optimization
* Infrastructure drift detection
* Agentic tool generation using Amazon Q

This is NOT:

* A chatbot
* A single-LLM wrapper
* A monitoring dashboard
* A simple alert engine

This IS:
A structured, deterministic, orchestrated multi-agent intelligence system.

---

# 🔷 Non-Negotiable Architectural Constraints

Copilot MUST follow these rules.

---

## 1️⃣ EXACT Agent Architecture (Mandatory)

The system MUST contain exactly six agents:

1. Observability_Agent
2. Infra_Agent
3. DB_Agent
4. Cost_Agent
5. CICD_Agent
6. Planner_Agent

No agent removal.
No renaming.
No merging responsibilities.

---

## 2️⃣ Orchestration Rule

* All inter-agent communication MUST go through Planner_Agent.
* Specialized agents MUST NOT call each other directly.
* Planner_Agent handles:

  * Task routing
  * Execution ordering
  * Response synthesis
  * Multi-agent coordination

---

## 3️⃣ Knowledge Base + RAG Requirement

The system MUST use:

* Bedrock Knowledge Bases
* Titan Embeddings
* S3 for storage

All agents MUST:

1. Retrieve relevant historical context
2. Perform reasoning using a well-reasoning LLM like Claude Sonnet, or etc.
3. Return structured responses

No prompt-only reasoning without context retrieval.

---

## 4️⃣ Amazon Q Tool Generation (Agentic Recursion)

System MUST support dynamic tool generation via Amazon Q.

ToolGenerator responsibilities:

* Infrastructure scanning tools
* ECS deployment pipelines
* IAM policy documents
* SQL migration scripts
* GitHub webhook listeners

Generated tools MUST:

* Be syntactically valid
* Include error handling
* Be validated before execution

---

## 5️⃣ Cost Awareness (India-First Rule)

All cost outputs MUST:

* Be in INR
* Include monthly projection
* Include annual projection
* Include confidence score

No USD values allowed in user-facing responses.

---

## 6️⃣ Structured Agent Communication Protocol

All agents MUST return:

```python
@dataclass
class AgentResponse:
    agent_type: str
    task_id: str
    status: str  # SUCCESS | PARTIAL | FAILED
    insights: List[Insight]
    data: Dict[str, Any]
    execution_time_ms: float
```

No free-text responses.
No unstructured LLM output.
All outputs must be parsed into structured models.

---

## 7️⃣ Semantic Insight Rule

All insights MUST include:

* Business impact context
* Severity level
* At least one actionable recommendation
* Cost impact (if applicable)

---

## 8️⃣ Predictive Analysis Rule

System MUST:

* Predict saturation when utilization > 80% trajectory
* Provide time-to-saturation estimate
* Rank recommendations by cost-benefit ratio

---

## 9️⃣ Correctness Properties (35 Required)

All implementation must satisfy defined correctness properties including:

* Agent routing correctness
* Orchestration ordering
* Knowledge base round-trip consistency
* Recommendation completeness
* Alert latency (< 60s for critical)
* Cost impact inclusion
* Commit attribution
* Tool generation validity

Property-based testing using Hypothesis is required.

Minimum:

* 100 iterations per property test

---

# 🔷 Technology Stack (Locked)

* Python 3.11+
* boto3
* Amazon Bedrock (Claude Sonnet)
* Bedrock Knowledge Bases
* Titan Embeddings
* Amazon Q Developer
* CloudWatch
* ECS
* AWS Billing API
* GitHub API
* Hypothesis (property testing)
* Pydantic or dataclasses
* IAM role-based authentication

---

# 🔷 Folder Structure (Mandatory)

```
autopilot-ai/
│
├── agents/
│   ├── planner_agent.py
│   ├── observability_agent.py
│   ├── infra_agent.py
│   ├── db_agent.py
│   ├── cost_agent.py
│   ├── cicd_agent.py
│
├── services/
│   ├── bedrock_client.py
│   ├── knowledge_base.py
│   ├── tool_generator.py
│   ├── cloudwatch_client.py
│   ├── billing_client.py
│   ├── github_client.py
│
├── models/
│   ├── core_models.py
│   ├── agent_protocol.py
│
├── tests/
│   ├── unit/
│   ├── property/
│
├── api/
│   ├── routes.py
│
└── main.py
```

Copilot must generate code consistent with this structure.

---

# 🔷 Development Phasing

Phase 1:

* Core data models
* Agent protocol
* Observability_Agent basic analysis
* Planner orchestration
* Bedrock integration

Phase 2:

* Knowledge Base RAG integration
* Cost Agent
* DB Agent

Phase 3:

* CI/CD Agent
* GitHub integration
* Amazon Q tool generation

Phase 4:

* Predictive analysis
* Alerting system
* Full property coverage

Copilot must NOT skip Planner implementation.

---

# 🔷 Design Philosophy

AutoPilotAI must be:

* Deterministic
* Modular
* Strictly typed
* Cost-aware
* Context-aware
* Safe
* Extensible
* Production-grade

Avoid:

* Demo-style scripts
* Hardcoded prompts in handlers
* Direct LLM calls without wrapper
* Cross-agent tight coupling
* Global state

---

# 🔷 Copilot Behavioral Instruction

When generating code:

* Always prefer modular design
* Always return structured JSON
* Always assume production deployment
* Always assume AWS IAM security constraints
* Always maintain agent separation
* Always integrate RAG before reasoning

If unsure:
Default to stricter architecture.

---

# 🔒 Architectural Lock Statement

Copilot must treat this project as:

A production multi-agent AI orchestration system with RAG, cost intelligence, predictive analytics, and dynamic tool generation — not a chatbot.