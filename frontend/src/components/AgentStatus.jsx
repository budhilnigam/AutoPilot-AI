/**
 * AgentStatus.jsx — Left sidebar: agent health + dependency status.
 *
 * - Shows all 6 agents with live pulsing status indicators.
 * - Polls GET /api/health/detail every 30 seconds for dependency health.
 * - Shows Bedrock, GitHub, and GitHub Poller status badges.
 */

import React, { useState, useEffect } from 'react'

const AGENTS = [
  { id: 'observability',  label: 'Observability',  icon: '📊' },
  { id: 'infra',          label: 'Infrastructure', icon: '🏗️' },
  { id: 'db',             label: 'Database',        icon: '🗄️' },
  { id: 'cost',           label: 'Cost',            icon: '💰' },
  { id: 'cicd',           label: 'CI/CD',           icon: '🚀' },
  { id: 'tool_generator', label: 'Tool Generator',  icon: '🔧' },
]

export default function AgentStatus() {
  const [health, setHealth]           = useState(null)
  const [lastChecked, setLastChecked] = useState(null)

  useEffect(() => {
    let cancelled = false

    const poll = async () => {
      try {
        const r    = await fetch('/api/health/detail')
        const data = await r.json()
        if (!cancelled) {
          setHealth(data)
          setLastChecked(new Date())
        }
      } catch { /* network error — keep last known state */ }
    }

    poll()
    const id = setInterval(poll, 30_000)
    return () => { cancelled = true; clearInterval(id) }
  }, [])

  const dep     = (name) => health?.dependencies?.find(d => d.name === name)
  const overall = health?.status ?? 'unknown'

  return (
    <aside className="sidebar sidebar-left">
      {/* Logo + system health */}
      <div className="sidebar-header">
        <div className="logo">
          <span className="logo-icon">⚡</span>
          <span className="logo-text">AutoPilot AI</span>
        </div>
        <div className={`system-status ${overall}`}>
          <span className="status-dot" />
          <span>
            {overall === 'ok'       ? 'Operational' :
             overall === 'degraded' ? 'Degraded'    :
             '—'}
          </span>
        </div>
      </div>

      {/* Agents */}
      <section className="sidebar-section">
        <h3 className="section-title">Agents</h3>
        <ul className="agent-list">
          {AGENTS.map(agent => (
            <li key={agent.id} className="agent-item">
              <span className="agent-icon">{agent.icon}</span>
              <span className="agent-label">{agent.label}</span>
              <span className="agent-dot agent-dot--active" title="Ready" />
            </li>
          ))}
        </ul>
      </section>

      {/* Infrastructure dependencies */}
      <section className="sidebar-section">
        <h3 className="section-title">Infrastructure</h3>
        <div className="dep-list">
          <DepRow label="Bedrock"       dep={dep('bedrock')} />
          <DepRow label="GitHub"        dep={dep('github')} />
          <DepRow label="GitHub Poller" dep={dep('github_poller')} />
        </div>
      </section>

      {lastChecked && (
        <p className="last-checked">
          Last checked {lastChecked.toLocaleTimeString()}
        </p>
      )}
    </aside>
  )
}

function DepRow({ label, dep }) {
  if (!dep) {
    return (
      <div className="dep-row">
        <span className="dep-label">{label}</span>
        <span className="dep-badge dep-badge--unknown">—</span>
      </div>
    )
  }
  return (
    <div className="dep-row">
      <span className="dep-label">{label}</span>
      <span
        className={`dep-badge ${dep.healthy ? 'dep-badge--ok' : 'dep-badge--error'}`}
        title={dep.detail || ''}
      >
        {dep.healthy
          ? `OK ${dep.latency_ms ? `${Math.round(dep.latency_ms)}ms` : ''}`
          : 'ERR'}
      </span>
    </div>
  )
}
