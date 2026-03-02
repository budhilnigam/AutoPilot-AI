"""
api/routes/alerts.py — Alert management endpoints + WebSocket live feed.

REST endpoints:
  GET  /api/alerts                — list recent alerts (ring buffer)
  GET  /api/alerts?severity=high  — filter by severity
  GET  /api/alerts/{alert_id}     — single alert lookup

WebSocket:
  WS   /api/alerts/ws             — live push feed; each new alert is sent
                                    as a JSON object the moment it fires.

WebSocket message format (server → client):
  {
    "type": "alert",
    "data": { ...Alert fields... }
  }

The WebSocket stays silent until an alert fires.  The connection is kept
alive via periodic ping frames (Starlette handles these automatically when
the ASGI server supports it).  On client disconnect the connection is
removed from the alert service's subscriber set.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from autopilot_ai.core.logging import get_logger
from autopilot_ai.models.domain import Alert, AlertSeverity
from autopilot_ai.services.alerting import alerting_service

logger = get_logger(__name__)
router = APIRouter(prefix="/api", tags=["alerts"])


# ── Response models ────────────────────────────────────────────────────────


class AlertListResponse(BaseModel):
    alerts: list[Alert]
    total: int


# ── REST routes ────────────────────────────────────────────────────────────


@router.get(
    "/alerts",
    response_model=AlertListResponse,
    summary="List recent alerts",
)
async def list_alerts(
    severity: str | None = Query(
        default=None,
        description="Filter by severity: critical, high, medium, low",
        pattern=r"^(critical|high|medium|low)$",
    ),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of alerts to return"),
) -> AlertListResponse:
    """
    Return alerts from the in-memory ring buffer, newest first.
    Optionally filter by severity.
    """
    if severity:
        try:
            sev_enum = AlertSeverity(severity)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Unknown severity: {severity!r}")
        alerts = alerting_service.get_by_severity(sev_enum)
        # Apply limit after filter
        alerts = alerts[-limit:]
    else:
        alerts = alerting_service.get_recent(limit)

    # Return newest first
    alerts = list(reversed(alerts))

    return AlertListResponse(alerts=alerts, total=len(alerts))


@router.get(
    "/alerts/{alert_id}",
    response_model=Alert,
    summary="Get a single alert by ID",
)
async def get_alert(alert_id: str) -> Alert:
    """Fetch a specific alert from the in-memory buffer by its alert_id."""
    for alert in alerting_service.get_recent(500):
        if alert.alert_id == alert_id:
            return alert
    raise HTTPException(status_code=404, detail=f"Alert {alert_id!r} not found.")


# ── WebSocket route ────────────────────────────────────────────────────────


@router.websocket("/alerts/ws")
async def alerts_websocket(ws: WebSocket) -> None:
    """
    Live alert feed via WebSocket.

    1. Accept the connection and register with AlertingService.
    2. Stay connected — AlertingService pushes JSON whenever an alert fires.
    3. On disconnect (client closes or network error), unregister cleanly.

    After connecting, the server immediately sends a snapshot of the most
    recent 20 alerts so the client can render an initial state without
    waiting for the next alert to fire.
    """
    await ws.accept()
    alerting_service.register_websocket(ws)

    logger.info("alerts_ws_connected", client=str(ws.client))

    try:
        # Send initial snapshot of recent alerts
        snapshot = alerting_service.get_recent(20)
        if snapshot:
            import json  # noqa: PLC0415
            await ws.send_text(json.dumps({
                "type": "snapshot",
                "data": [a.model_dump(mode="json") for a in reversed(snapshot)],
            }))

        # Keep the connection alive by consuming incoming frames.
        # Clients may send ping/close frames; we handle them here.
        while True:
            try:
                # recv_text with a long timeout so we don't busy-loop
                # We discard client messages (this is a one-way push feed).
                await ws.receive_text()
            except WebSocketDisconnect:
                break

    finally:
        alerting_service.unregister_websocket(ws)
        logger.info("alerts_ws_disconnected", client=str(ws.client))
