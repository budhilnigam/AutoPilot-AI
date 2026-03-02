/**
 * useSSEStream.js — SSE streaming hook for POST /api/query/stream
 *
 * Returns:
 *   messages      — array of { id, role, content, status, agentProgress }
 *   isStreaming    — true while a request is in-flight
 *   sendQuery(q)  — fire a new query (cancels any in-flight request)
 *   clearMessages — wipe the chat history
 *
 * SSE event handling:
 *   event: agent_progress  → appends a pill to message.agentProgress
 *   event: done            → sets message.content to the narrative
 *   event: error           → sets message.content to the error text
 *   ": keep-alive"         → ignored (comment line)
 */

import { useState, useRef, useCallback } from 'react'

const AGENT_STATUS_COLORS = {
  success: '#22d3ee',
  partial: '#fbbf24',
  failed:  '#f43f5e',
  timeout: '#fb923c',
  skipped: '#50506a',
}

export function useSSEStream() {
  const [messages, setMessages]     = useState([])
  const [isStreaming, setIsStreaming] = useState(false)
  const abortRef = useRef(null)

  /** Parse the raw SSE byte stream into discrete events. */
  function* parseSseChunk(buffer, incoming) {
    buffer += incoming
    const parts = buffer.split('\n\n')
    // Everything except the last part is a complete event
    for (let i = 0; i < parts.length - 1; i++) {
      const block = parts[i]
      let eventType = 'message'
      let data = ''
      for (const line of block.split('\n')) {
        if (line.startsWith('event: '))      eventType = line.slice(7).trim()
        else if (line.startsWith('data: '))  data = line.slice(6).trim()
        else if (line.startsWith(': '))      { /* comment / keep-alive */ }
      }
      if (data) yield { type: eventType, data }
    }
    return parts[parts.length - 1]  // return the incomplete tail
  }

  const sendQuery = useCallback(async (query, context = {}, mode = 'query') => {
    // Cancel any in-flight stream
    if (abortRef.current) abortRef.current.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl

    const userMsgId = crypto.randomUUID()
    const aiMsgId   = crypto.randomUUID()

    setMessages(prev => [
      ...prev,
      { id: userMsgId, role: 'user',      content: query },
      { id: aiMsgId,   role: 'assistant', content: '', status: 'streaming', agentProgress: [] },
    ])
    setIsStreaming(true)

    let aborted = false

    try {
      const resp = await fetch('/api/query/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, context, mode }),
        signal: ctrl.signal,
      })

      if (!resp.ok) {
        const errBody = await resp.json().catch(() => ({}))
        throw new Error(errBody.message || `HTTP ${resp.status}`)
      }

      const reader  = resp.body.getReader()
      const decoder = new TextDecoder()
      let buffer    = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        const chunk = decoder.decode(value, { stream: true })

        // Iterate over complete SSE events extracted from the chunk
        const gen = parseSseChunk(buffer, chunk)
        while (true) {
          const result = gen.next()
          if (result.done) {
            buffer = result.value  // leftover incomplete block
            break
          }
          const { type, data } = result.value
          handleEvent(type, data, aiMsgId)
        }
      }

    } catch (err) {
      if (err.name === 'AbortError') {
        aborted = true
        // Remove the empty placeholder — the user message stays in history
        setMessages(prev => prev.filter(m => m.id !== aiMsgId))
        return
      }
      setMessages(prev => prev.map(m =>
        m.id === aiMsgId
          ? { ...m, content: `⚠️ ${err.message}`, status: 'error' }
          : m,
      ))
    } finally {
      if (abortRef.current === ctrl) setIsStreaming(false)
      if (!aborted) {
        // If the stream ended without a 'done' event, close gracefully
        setMessages(prev => prev.map(m =>
          m.id === aiMsgId && m.status === 'streaming'
            ? { ...m, status: 'done' }
            : m,
        ))
      }
    }
  }, [])

  function handleEvent(type, rawData, aiMsgId) {
    let payload
    try { payload = JSON.parse(rawData) }
    catch { return }

    if (type === 'agent_progress') {
      setMessages(prev => prev.map(m =>
        m.id === aiMsgId
          ? { ...m, agentProgress: [...(m.agentProgress || []), { ...payload, _color: AGENT_STATUS_COLORS[payload.status] }] }
          : m,
      ))
    } else if (type === 'done') {
      setMessages(prev => prev.map(m =>
        m.id === aiMsgId
          ? { ...m, content: payload.narrative || '*(No narrative)*', status: 'done', queryId: payload.query_id }
          : m,
      ))
    } else if (type === 'error') {
      setMessages(prev => prev.map(m =>
        m.id === aiMsgId
          ? { ...m, content: `⚠️ ${payload.message || 'An error occurred.'}`, status: 'error' }
          : m,
      ))
    }
  }

  const clearMessages = useCallback(() => setMessages([]), [])

  return { messages, isStreaming, sendQuery, clearMessages }
}
