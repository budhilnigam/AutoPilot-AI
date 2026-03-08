"""Quick test for planner keyword fallback routing."""
from autopilot_ai.agents.planner import PlannerAgent

p = PlannerAgent()

tests = [
    "How much am I spending on AWS?",
    "List my repos",
    "What is my infrastructure?",
    "CPU usage is high",
    "CI quota remaining from GitHub",
    "Show my AWS costs",
    "What are my latest commits",
    "Give me an infrastructure description",
    "Any anomalies in my metrics?",
    "Show my database performance",
]

print("Keyword fallback routing test:")
print("-" * 60)
for q in tests:
    plans = p._keyword_fallback_route(q, "query")
    agents = [f"{p['agent_type']}:{p['task_type']}" for p in plans]
    status = "OK" if agents else "EMPTY"
    print(f"  [{status}] {q}")
    for a in agents:
        print(f"         -> {a}")
print("-" * 60)
print("Done")
