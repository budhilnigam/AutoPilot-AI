/**
 * AlertFeed.jsx — Right sidebar: live WebSocket alert feed.
 *
 * - Connects to WS /api/alerts/ws via useAlertSocket.
 * - Shows an initial snapshot of recent alerts on connect.
 * - New alerts fly in from the right with an animation.
 * - Click "Investigate →" to open the alert in the chat.
 * - Filter bar: all / critical / high / medium / low.
 */

import React, { useState } from 'react'
import { useAlertSocket } from '../hooks/useAlertSocket'

const SEVERITY_CFG = {
  critical: { color: 'var(--critical)', icon: '🔴', label: 'CRITICAL' },
  high:     { color: 'var(--high)',     icon: '🟠', label: 'HIGH'     },
  medium:   { color: 'var(--medium)',   icon: '🟡', label: 'MEDIUM'   },
  low:      { color: 'var(--low)',      icon: '⚪', label: 'LOW'      },
}

function formatTime(iso) {
  if (!iso) return ''
  return new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
}

function formatInr(n) {
  if (!n) return null
  return `₹${Math.round(n).toLocaleString('en-IN')}/mo`
}

export default function AlertFeed({ onAlertClick }) {
  const { alerts, status, dismissAlert } = useAlertSocket()
  const [filter, setFilter] = useState('all')

  const filtered = filter === 'all'
    ? alerts
    : alerts.filter(a => a.severity === filter)

  return (
    <aside className="sidebar sidebar-right">
      {/* Header */}
      <div className="sidebar-header">
        <h3>Live Alerts</h3>
        <span className={`ws-badge ${status}`}>
          {status === 'connected'    && '● LIVE'}
          {status === 'connecting'   && '◌ ...'}
          {status === 'disconnected' && '○ OFF'}
        </span>
      </div>

      {/* Severity filter */}
      <div className="alert-filters">
        {['all', 'critical', 'high', 'medium', 'low'].map(f => (
          <button
            key={f}
            className={`filter-btn${filter === f ? ' active' : ''}`}
            onClick={() => setFilter(f)}
          >
            {f}
          </button>
        ))}
      </div>

      {/* Alert list */}
      <div className="alert-list">
        {filtered.length === 0 ? (
          <div className="alert-empty">
            No {filter !== 'all' ? filter + ' ' : ''}alerts
          </div>
        ) : (
          filtered.map(alert => (
            <AlertCard
              key={alert.alert_id}
              alert={alert}
              onDrillDown={() => onAlertClick?.(alert)}
              onDismiss={() => dismissAlert(alert.alert_id)}
            />
          ))
        )}
      </div>
    </aside>
  )
}

function AlertCard({ alert, onDrillDown, onDismiss }) {
  const cfg  = SEVERITY_CFG[alert.severity] || SEVERITY_CFG.low
  const cost = formatInr(alert.cost_impact_inr)

  return (
    <div className="alert-card" style={{ borderLeftColor: cfg.color }}>
      <div className="alert-card-header">
        <span className="alert-severity" style={{ color: cfg.color }}>
          {cfg.icon} {cfg.label}
        </span>
        <span className="alert-time">{formatTime(alert.detected_at)}</span>
      </div>

      <p className="alert-component">{alert.component}</p>
      <p className="alert-title">{alert.title}</p>

      {cost && <span className="alert-cost">{cost} impact</span>}

      {alert.recommendations?.length > 0 && (
        <p className="alert-rec">→ {alert.recommendations[0]}</p>
      )}

      <div className="alert-card-actions">
        <button className="btn-drill-down" onClick={onDrillDown}>
          Investigate →
        </button>
        <button className="btn-dismiss" onClick={onDismiss} aria-label="Dismiss">
          ✕
        </button>
      </div>
    </div>
  )
}
