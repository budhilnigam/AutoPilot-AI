import React, { useState } from 'react'
import AgentStatus from './components/AgentStatus'
import ChatPanel from './components/ChatPanel'
import AlertFeed from './components/AlertFeed'

export default function App() {
  // When the user clicks "Investigate →" on an alert card, this state
  // pre-loads it into the chat as an alert-mode query.
  const [drillDownQuery, setDrillDownQuery] = useState(null)

  const handleAlertClick = (alert) => {
    setDrillDownQuery({
      // Timestamp ensures a unique object even if user clicks same alert twice
      _ts: Date.now(),
      query:
        `Investigate this alert: ${alert.title}\n\n` +
        `Component: ${alert.component}\n` +
        `Severity: ${alert.severity}\n` +
        `Description: ${alert.description || alert.title}`,
      context: {
        alert_id: alert.alert_id,
        severity: alert.severity,
        component: alert.component,
        commit_sha: alert.commit_sha || null,
        mode: 'alert',
      },
    })
  }

  return (
    <div className="app-layout">
      <AgentStatus />
      <ChatPanel
        initialQuery={drillDownQuery}
        onQuerySent={() => setDrillDownQuery(null)}
      />
      <AlertFeed onAlertClick={handleAlertClick} />
    </div>
  )
}
