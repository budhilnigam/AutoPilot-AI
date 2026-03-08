"""
Unified Agent

Single strong agent powered by dynamic AWS SDK tool-calling.
This consolidates prior specialized agent responsibilities.
"""

import logging
from typing import Any, Dict

from models.agent_protocol import AgentType
from .observability_agent import ObservabilityAgent

logger = logging.getLogger(__name__)


class UnifiedAgent(ObservabilityAgent):
    """Unified all-domain agent built on dynamic AWS tool execution."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.agent_type = AgentType.UNIFIED
        logger.info("Unified Agent initialized (single-agent mode)")

    def process_task(self, task_id: str, query: str, context: Dict[str, Any]):
        """Compatibility shim for planner execution path."""
        return self.analyze_with_dynamic_tools(task_id=task_id, user_query=query, context=context)
