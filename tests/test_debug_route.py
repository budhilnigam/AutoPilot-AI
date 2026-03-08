"""Quick debug: what does the LLM planner actually pass to the github agent?"""
import asyncio, os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

async def main():
    from autopilot_ai.agents.planner import PlannerAgent
    p = PlannerAgent()
    plans = await p._route_query("List my repositories", {}, "query")
    for plan in plans:
        print(f"Agent: {plan.get('agent_type')} / {plan.get('task_type')}")
        print(f"Params: {plan.get('parameters')}")

asyncio.run(main())
