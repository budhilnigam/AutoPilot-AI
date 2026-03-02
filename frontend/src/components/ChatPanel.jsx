/**
 * ChatPanel.jsx — Center panel: streaming chat interface.
 *
 * - Sends queries to POST /api/query/stream via the useSSEStream hook.
 * - Shows per-agent progress pills as each agent finishes.
 * - Renders AI responses as markdown.
 * - Accepts `initialQuery` prop for alert drill-down (pre-loaded from
 *   AlertFeed when user clicks "Investigate →").
 */

import React, { useState, useRef, useEffect } from 'react'
import MessageBubble from './MessageBubble'
import { useSSEStream } from '../hooks/useSSEStream'

const SUGGESTIONS = [
  'Why is checkout slow right now?',
  'What is our biggest AWS cost driver this month?',
  'Are there any database query bottlenecks?',
  'Give me a full system health snapshot.',
  'Show the top 3 CI/CD regressions from the last week.',
]

export default function ChatPanel({ initialQuery, onQuerySent }) {
  const [input, setInput] = useState('')
  const { messages, isStreaming, sendQuery, clearMessages } = useSSEStream()
  const bottomRef  = useRef(null)
  const inputRef   = useRef(null)

  // Auto-scroll to bottom when messages change
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  // Alert drill-down: when parent passes a new initialQuery, fire it
  useEffect(() => {
    if (!initialQuery) return
    sendQuery(initialQuery.query, initialQuery.context || {}, 'alert')
    onQuerySent?.()
  }, [initialQuery])  // eslint-disable-line react-hooks/exhaustive-deps

  const handleSubmit = (e) => {
    e.preventDefault()
    const q = input.trim()
    if (!q || isStreaming) return
    setInput('')
    sendQuery(q)
  }

  const handleKeyDown = (e) => {
    // Shift+Enter adds a newline (future: textarea). Enter submits.
    if (e.key === 'Enter' && !e.shiftKey) handleSubmit(e)
  }

  return (
    <main className="chat-panel">
      {/* Header */}
      <div className="chat-header">
        <h2>SRE Copilot</h2>
        {messages.length > 0 && (
          <button className="btn-ghost" onClick={clearMessages} disabled={isStreaming}>
            Clear
          </button>
        )}
      </div>

      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 ? (
          <div className="chat-empty">
            <div className="chat-empty-icon">⚡</div>
            <h3>Ask AutoPilot AI anything</h3>
            <p>Metrics · Costs · Infrastructure · Database · CI/CD</p>
            <div className="suggestions">
              {SUGGESTIONS.map((s, i) => (
                <button
                  key={i}
                  className="suggestion-chip"
                  onClick={() => !isStreaming && sendQuery(s)}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map(msg => <MessageBubble key={msg.id} message={msg} />)
        )}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <form className="chat-input-row" onSubmit={handleSubmit}>
        <input
          ref={inputRef}
          className="chat-input"
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={isStreaming ? 'Analysing your infrastructure…' : 'Ask anything about your infrastructure…'}
          disabled={isStreaming}
          autoComplete="off"
          autoFocus
        />
        <button
          className="btn-send"
          type="submit"
          disabled={!input.trim() || isStreaming}
          aria-label="Send"
        >
          {isStreaming ? (
            <span className="spinner" />
          ) : (
            <svg viewBox="0 0 24 24" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 2L11 13M22 2L15 22 11 13 2 9l20-7z" />
            </svg>
          )}
        </button>
      </form>
    </main>
  )
}
