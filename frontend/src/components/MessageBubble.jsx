/**
 * MessageBubble.jsx — Renders a single user or assistant message.
 *
 * User messages:  plain text in a gradient bubble, right-aligned.
 * AI messages:    avatar + agent progress pills + markdown narrative,
 *                 with a typing indicator while streaming.
 */

import React from 'react'
import Markdown from 'react-markdown'

const AGENT_ICONS = {
  observability:  '📊',
  infra:          '🏗️',
  db:             '🗄️',
  cost:           '💰',
  cicd:           '🚀',
  planner:        '🤖',
}

export default function MessageBubble({ message }) {
  const { role, content, status, agentProgress, diagnostics } = message

  if (role === 'user') {
    return (
      <div className="message message--user">
        <div className="message-bubble">{content}</div>
      </div>
    )
  }

  // ── Assistant message ─────────────────────────────────────────────────────
  const isStreaming = status === 'streaming'
  const isError     = status === 'error'
  const hasDiagnostics = diagnostics?.has_failures && Array.isArray(diagnostics?.items)

  return (
    <div className="message message--assistant">
      <div className="message-avatar">⚡</div>

      <div className="message-body">
        {/* Per-agent progress pills (appear as each agent finishes) */}
        {agentProgress && agentProgress.length > 0 && (
          <div className="agent-progress-bar">
            {agentProgress.map((ap, i) => (
              <span
                key={i}
                className="agent-pill"
                title={
                  ap.error
                    ? `${ap.agent}: ${ap.error}${ap.error_hint ? `\nHint: ${ap.error_hint}` : ''}`
                    : `${ap.agent}: ${ap.insight_count ?? 0} insight${ap.insight_count !== 1 ? 's' : ''} · ${ap.execution_time_ms ?? 0}ms`
                }
                style={{ borderColor: ap._color || 'var(--text-muted)' }}
              >
                <span>{AGENT_ICONS[ap.agent] || '⚙️'}</span>
                <span>{ap.insight_count ?? 0}</span>
              </span>
            ))}
          </div>
        )}

        {/* Bubble */}
        <div className={`message-bubble message-bubble--assistant${isError ? ' message-bubble--error' : ''}`}>
          {isStreaming && !content ? (
            /* No content yet → three-dot loader */
            <div className="typing-indicator">
              <span /><span /><span />
            </div>
          ) : (
            <Markdown>{content || ''}</Markdown>
          )}
          {isStreaming && content && (
            <span className="cursor-blink">▋</span>
          )}
        </div>

        {hasDiagnostics && (
          <div className="message-diagnostics">
            <div className="message-diagnostics-title">Tool/Service Failures</div>
            {diagnostics.items.map((item, idx) => (
              <div key={idx} className="message-diagnostic-item">
                <strong>{item.agent}</strong>
                {` (${item.status})`}
                {`: ${item.category}`}
                <div>{item.hint}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
