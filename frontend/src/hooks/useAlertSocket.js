/**
 * useAlertSocket.js — WebSocket hook for WS /api/alerts/ws
 *
 * Returns:
 *   alerts        — array of Alert objects (newest first)
 *   status        — 'connecting' | 'connected' | 'disconnected'
 *   dismissAlert  — (alertId) => void
 *
 * On connect, the server sends an initial snapshot of recent alerts
 * ({ type: 'snapshot', data: [...] }), then individual alerts as they fire
 * ({ type: 'alert', data: {...} }).
 *
 * Auto-reconnects with a 3-second delay on close/error.
 */

import { useState, useEffect, useRef, useCallback } from 'react'

const MAX_ALERTS = 100

function getWsUrl() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
  return `${protocol}//${window.location.host}/api/alerts/ws`
}

export function useAlertSocket() {
  const [alerts, setAlerts]   = useState([])
  const [status, setStatus]   = useState('disconnected')
  const wsRef                 = useRef(null)
  const reconnectTimer        = useRef(null)
  const mountedRef            = useRef(true)

  const connect = useCallback(() => {
    if (!mountedRef.current) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')
    let ws
    try {
      ws = new WebSocket(getWsUrl())
    } catch {
      // WebSocket URL invalid (e.g. no host). Retry later.
      reconnectTimer.current = setTimeout(connect, 5000)
      setStatus('disconnected')
      return
    }
    wsRef.current = ws

    ws.onopen = () => {
      if (!mountedRef.current) return
      setStatus('connected')
      clearTimeout(reconnectTimer.current)
    }

    ws.onmessage = (event) => {
      if (!mountedRef.current) return
      try {
        const msg = JSON.parse(event.data)
        if (msg.type === 'snapshot') {
          // Server sends newest-first; we keep that order
          setAlerts((msg.data || []).slice(0, MAX_ALERTS))
        } else if (msg.type === 'alert') {
          setAlerts(prev => [msg.data, ...prev].slice(0, MAX_ALERTS))
        }
      } catch { /* ignore malformed frames */ }
    }

    ws.onclose = () => {
      if (!mountedRef.current) return
      setStatus('disconnected')
      reconnectTimer.current = setTimeout(connect, 3000)
    }

    ws.onerror = () => {
      ws.close()
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    connect()
    return () => {
      mountedRef.current = false
      clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  const dismissAlert = useCallback((alertId) => {
    setAlerts(prev => prev.filter(a => a.alert_id !== alertId))
  }, [])

  return { alerts, status, dismissAlert }
}
