"""
services/alerting.py — Alert creation, deduplication, and delivery.

Responsibilities:
  create_alert(insight, context)  → build an Alert from an agent Insight
  process_alerts(responses)       → extract + dedup + broadcast alerts from
                                     a list of AgentResponses
  broadcast(alert)                → push alert to all connected WebSocket clients
  get_recent_alerts(n)            → return last N alerts (in-memory ring buffer)

Deduplication (Property 29):
  An alert is suppressed if an alert with the same dedup_key was created
  within the last 5 minutes. dedup_key = SHA-256(severity + component + title,
  truncated to 5-min bucket).

Delivery SLA (Property 28):
  CRITICAL alerts must be delivered (broadcast) within 60 seconds of detection.
  Delivery latency is logged on every broadcast so it is observable.

WebSocket broadcast:
  The alerting service holds a set of active WebSocket connections registered
  by the API layer. When an alert fires, it is serialised to JSON and sent
  to all connected clients. Disconnected clients are removed from the set.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING, Any

from autopilot_ai.core.config import settings
from autopilot_ai.core.logging import get_logger
from autopilot_ai.models.domain import Alert, AlertSeverity
from autopilot_ai.models.insights import Insight, Urgency
from autopilot_ai.models.responses import AgentResponse

if TYPE_CHECKING:
    from starlette.websockets import WebSocket

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
_DEDUP_WINDOW_SECONDS = 300          # 5-minute dedup window
_RING_BUFFER_SIZE = 500              # in-memory recent-alerts buffer
_CRITICAL_SLA_SECONDS = 60          # Property 28


# ── Urgency → Severity mapping ────────────────────────────────────────────────
_URGENCY_TO_SEVERITY: dict[Urgency, AlertSeverity] = {
    Urgency.IMMEDIATE: AlertSeverity.CRITICAL,
    Urgency.HIGH: AlertSeverity.HIGH,
    Urgency.MEDIUM: AlertSeverity.MEDIUM,
    Urgency.LOW: AlertSeverity.LOW,
}


def _dedup_key(severity: AlertSeverity, component: str, title: str) -> str:
    """
    Stable dedup key: SHA-256 of (severity + component + title + 5-min bucket).
    Two alerts with identical severity/component/title within the same 5-min
    window hash to the same key and are suppressed.
    """
    now_bucket = int(time.time() // _DEDUP_WINDOW_SECONDS)
    raw = f"{severity.value}:{component}:{title}:{now_bucket}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _insight_to_alert(insight: Insight, commit_sha: str | None = None) -> Alert:
    """Convert an Insight into an Alert, computing severity and dedup_key."""
    severity = _URGENCY_TO_SEVERITY.get(insight.urgency, AlertSeverity.MEDIUM)
    dk = _dedup_key(severity, insight.component, insight.title)

    rec_texts = [r.action for r in insight.recommendations]
    best_cost = max(
        (r.cost_impact.monthly_inr for r in insight.recommendations if r.cost_impact),
        default=None,
    )

    return Alert(
        alert_id=str(uuid.uuid4()),
        severity=severity,
        title=insight.title,
        description=insight.business_context or insight.title,
        component=insight.component,
        detected_at=datetime.now(tz=timezone.utc),
        cost_impact_inr=best_cost,
        commit_sha=commit_sha,
        recommendations=rec_texts,
        dedup_key=dk,
    )


class AlertingService:
    """
    Central alert bus.

    - Receives AgentResponse objects, extracts high-urgency Insights, and
      converts them to Alerts.
    - Deduplicates within a 5-minute window.
    - Broadcasts to all registered WebSocket connections.
    - Maintains a ring buffer of recent alerts for the GET /api/alerts endpoint.
    """

    def __init__(self) -> None:
        self._recent: deque[Alert] = deque(maxlen=_RING_BUFFER_SIZE)
        self._dedup_seen: dict[str, float] = {}   # key → epoch of last seen
        self._websockets: set["WebSocket"] = set()
        self._lock = asyncio.Lock()

    # ── WebSocket registration ────────────────────────────────────────────

    def register_websocket(self, ws: "WebSocket") -> None:
        self._websockets.add(ws)
        logger.info("alerting_ws_registered", total=len(self._websockets))

    def unregister_websocket(self, ws: "WebSocket") -> None:
        self._websockets.discard(ws)
        logger.info("alerting_ws_unregistered", total=len(self._websockets))

    # ── Public API ────────────────────────────────────────────────────────

    async def process_responses(
        self,
        responses: list[AgentResponse],
        commit_sha: str | None = None,
    ) -> list[Alert]:
        """
        Extract Insights from AgentResponses, convert HIGH/IMMEDIATE urgency
        ones to Alerts, dedup, and broadcast.

        Returns the list of new (non-deduped) alerts fired.
        """
        fired: list[Alert] = []

        for resp in responses:
            for insight in resp.insights:
                if insight.urgency not in (Urgency.IMMEDIATE, Urgency.HIGH):
                    continue  # only alert on high/immediate urgency

                alert = _insight_to_alert(insight, commit_sha=commit_sha)

                if await self._is_duplicate(alert):
                    logger.debug(
                        "alerting_suppressed_duplicate",
                        dedup_key=alert.dedup_key,
                        title=alert.title,
                    )
                    continue

                await self._store_and_broadcast(alert)
                fired.append(alert)

        return fired

    async def create_alert(
        self,
        insight: Insight,
        commit_sha: str | None = None,
    ) -> Alert | None:
        """
        Create a single alert from an Insight (used by GitHub poller and
        direct callers). Returns None if the alert is deduplicated.
        """
        alert = _insight_to_alert(insight, commit_sha=commit_sha)

        if await self._is_duplicate(alert):
            logger.debug("alerting_suppressed_duplicate", title=alert.title)
            return None

        await self._store_and_broadcast(alert)
        return alert

    def get_recent(self, n: int = 50) -> list[Alert]:
        """Return the last `n` alerts from the in-memory ring buffer."""
        alerts = list(self._recent)
        return alerts[-n:]

    def get_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        return [a for a in self._recent if a.severity == severity]

    # ── Internal helpers ──────────────────────────────────────────────────

    async def _is_duplicate(self, alert: Alert) -> bool:
        """Return True if this dedup_key was seen within _DEDUP_WINDOW_SECONDS."""
        if alert.dedup_key is None:
            return False
        async with self._lock:
            last_seen = self._dedup_seen.get(alert.dedup_key)
            now = time.monotonic()
            if last_seen and (now - last_seen) < _DEDUP_WINDOW_SECONDS:
                return True
            self._dedup_seen[alert.dedup_key] = now
            # Prune stale entries to avoid unbounded growth
            cutoff = now - _DEDUP_WINDOW_SECONDS * 2
            self._dedup_seen = {k: v for k, v in self._dedup_seen.items() if v > cutoff}
            return False

    async def _store_and_broadcast(self, alert: Alert) -> None:
        """Append to ring buffer and push to WebSocket clients."""
        self._recent.append(alert)

        # Log delivery for SLA tracking (Property 28)
        alert = alert.model_copy(update={"delivered_at": datetime.now(tz=timezone.utc)})
        latency = alert.delivery_latency_seconds or 0
        log_fn = logger.warning if (
            alert.severity == AlertSeverity.CRITICAL and latency > _CRITICAL_SLA_SECONDS
        ) else logger.info

        log_fn(
            "alerting_fired",
            alert_id=alert.alert_id,
            severity=alert.severity.value,
            title=alert.title,
            component=alert.component,
            delivery_latency_s=round(latency, 3),
            cost_impact_inr=alert.cost_impact_inr,
            sla_breach=latency > _CRITICAL_SLA_SECONDS,
        )

        await self._broadcast(alert)

    async def _broadcast(self, alert: Alert) -> None:
        """Send alert JSON to all connected WebSocket clients."""
        if not self._websockets:
            return

        payload = json.dumps({
            "type": "alert",
            "data": alert.model_dump(mode="json"),
        })

        dead: set["WebSocket"] = set()
        for ws in list(self._websockets):
            try:
                await ws.send_text(payload)
            except Exception as e:
                logger.debug("alerting_ws_send_failed", error=str(e)[:60])
                dead.add(ws)

        for ws in dead:
            self.unregister_websocket(ws)


# Module-level singleton
alerting_service = AlertingService()
