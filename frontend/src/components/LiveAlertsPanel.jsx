import React, { useState, useEffect, useRef } from 'react'
import { 
  Bell, 
  AlertTriangle, 
  AlertCircle, 
  Info,
  TrendingUp,
  DollarSign,
  Activity,
  X
} from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000'

const severityConfig = {
  critical: {
    icon: AlertTriangle,
    color: 'text-red-600',
    bgColor: 'bg-red-50',
    borderColor: 'border-red-200',
  },
  high: {
    icon: AlertCircle,
    color: 'text-orange-600',
    bgColor: 'bg-orange-50',
    borderColor: 'border-orange-200',
  },
  medium: {
    icon: AlertCircle,
    color: 'text-yellow-600',
    bgColor: 'bg-yellow-50',
    borderColor: 'border-yellow-200',
  },
  low: {
    icon: Info,
    color: 'text-blue-600',
    bgColor: 'bg-blue-50',
    borderColor: 'border-blue-200',
  },
  info: {
    icon: Info,
    color: 'text-gray-600',
    bgColor: 'bg-gray-50',
    borderColor: 'border-gray-200',
  },
}

function LiveAlertsPanel() {
  const [alerts, setAlerts] = useState([])
  const [connected, setConnected] = useState(false)
  const [filter, setFilter] = useState('all') // all, critical, high, medium, low
  const wsRef = useRef(null)

  useEffect(() => {
    connectWebSocket()
    
    // Add some mock alerts for demonstration
    const mockAlerts = [
      {
        id: 1,
        severity: 'critical',
        title: 'CPU Utilization Critical',
        message: 'ECS service "api-production" CPU > 90% for 5 minutes',
        source: 'CloudWatch',
        timestamp: new Date(Date.now() - 2 * 60 * 1000),
        costImpact: '₹45,000/month if not addressed',
      },
      {
        id: 2,
        severity: 'high',
        title: 'Database Query Slow',
        message: 'Query execution time increased by 300% in last hour',
        source: 'Database Agent',
        timestamp: new Date(Date.now() - 15 * 60 * 1000),
        recommendation: 'Consider adding index on users.email column',
      },
      {
        id: 3,
        severity: 'medium',
        title: 'Build Time Regression',
        message: 'Build #342 took 12m (baseline: 8m, +50%)',
        source: 'CI/CD Agent',
        timestamp: new Date(Date.now() - 30 * 60 * 1000),
        commit: 'abc123f',
      },
      {
        id: 4,
        severity: 'low',
        title: 'Cost Optimization Available',
        message: 'EC2 instance "worker-01" utilization < 30% for 7 days',
        source: 'Cost Agent',
        timestamp: new Date(Date.now() - 60 * 60 * 1000),
        costImpact: 'Save ₹18,300/month with right-sizing',
      },
    ]
    
    setAlerts(mockAlerts)

    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  const connectWebSocket = () => {
    try {
      const ws = new WebSocket(`${WS_URL}/ws/alerts`)
      
      ws.onopen = () => {
        console.log('WebSocket connected')
        setConnected(true)
        
        // Send ping to keep connection alive
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping')
          }
        }, 30000)
        
        ws.pingInterval = pingInterval
      }
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          
          if (data.type === 'alert') {
            const newAlert = {
              id: Date.now(),
              ...data.data,
              timestamp: new Date(data.timestamp),
            }
            
            setAlerts(prev => [newAlert, ...prev].slice(0, 50)) // Keep last 50 alerts
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error)
        }
      }
      
      ws.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnected(false)
      }
      
      ws.onclose = () => {
        console.log('WebSocket disconnected')
        setConnected(false)
        
        if (ws.pingInterval) {
          clearInterval(ws.pingInterval)
        }
        
        // Attempt to reconnect after 5 seconds
        setTimeout(connectWebSocket, 5000)
      }
      
      wsRef.current = ws
    } catch (error) {
      console.error('Failed to connect WebSocket:', error)
    }
  }

  const dismissAlert = (alertId) => {
    setAlerts(prev => prev.filter(a => a.id !== alertId))
  }

  const filteredAlerts = filter === 'all' 
    ? alerts 
    : alerts.filter(a => a.severity === filter)

  const Alert = ({ alert }) => {
    const config = severityConfig[alert.severity] || severityConfig.info
    const Icon = config.icon

    return (
      <div className={`p-4 border rounded-lg ${config.bgColor} ${config.borderColor} transition-all hover:shadow-md`}>
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-3 flex-1">
            <Icon className={`w-5 h-5 ${config.color} mt-0.5 flex-shrink-0`} />
            
            <div className="flex-1 min-w-0">
              <div className="flex items-start justify-between">
                <h4 className="text-sm font-semibold text-gray-900">{alert.title}</h4>
                <button
                  onClick={() => dismissAlert(alert.id)}
                  className="ml-2 p-1 hover:bg-gray-200 rounded transition-colors"
                  title="Dismiss"
                >
                  <X className="w-4 h-4 text-gray-500" />
                </button>
              </div>
              
              <p className="text-sm text-gray-700 mt-1">{alert.message}</p>
              
              {alert.recommendation && (
                <div className="mt-2 p-2 bg-white rounded border border-gray-200">
                  <p className="text-xs text-gray-600">
                    <span className="font-medium">Recommendation:</span> {alert.recommendation}
                  </p>
                </div>
              )}
              
              {alert.costImpact && (
                <div className="mt-2 flex items-center text-xs text-gray-600">
                  <DollarSign className="w-3 h-3 mr-1" />
                  {alert.costImpact}
                </div>
              )}
              
              <div className="mt-2 flex items-center justify-between text-xs text-gray-500">
                <span>{alert.source}</span>
                <span>{formatDistanceToNow(alert.timestamp, { addSuffix: true })}</span>
              </div>
              
              {alert.commit && (
                <div className="mt-1 text-xs text-gray-500">
                  Commit: <code className="px-1 bg-gray-200 rounded">{alert.commit}</code>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  const severityCount = (severity) => {
    return alerts.filter(a => a.severity === severity).length
  }

  return (
    <div className="p-4 h-full flex flex-col">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-bold text-gray-900 flex items-center">
            <Bell className="w-5 h-5 mr-2" />
            Live Alerts
          </h2>
          
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${connected ? 'bg-green-500' : 'bg-gray-400'}`} />
            <span className="text-xs text-gray-600">
              {connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
        </div>

        {/* Filter Buttons */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => setFilter('all')}
            className={`px-3 py-1 text-xs rounded-full transition-colors ${
              filter === 'all'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            All ({alerts.length})
          </button>
          <button
            onClick={() => setFilter('critical')}
            className={`px-3 py-1 text-xs rounded-full transition-colors ${
              filter === 'critical'
                ? 'bg-red-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Critical ({severityCount('critical')})
          </button>
          <button
            onClick={() => setFilter('high')}
            className={`px-3 py-1 text-xs rounded-full transition-colors ${
              filter === 'high'
                ? 'bg-orange-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            High ({severityCount('high')})
          </button>
          <button
            onClick={() => setFilter('medium')}
            className={`px-3 py-1 text-xs rounded-full transition-colors ${
              filter === 'medium'
                ? 'bg-yellow-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            Medium ({severityCount('medium')})
          </button>
        </div>
      </div>

      {/* Alerts List */}
      <div className="flex-1 overflow-y-auto scrollbar-thin space-y-3">
        {filteredAlerts.length === 0 ? (
          <div className="text-center py-12">
            <Bell className="w-12 h-12 text-gray-300 mx-auto mb-3" />
            <p className="text-sm text-gray-500">No alerts to display</p>
            <p className="text-xs text-gray-400 mt-1">
              {filter !== 'all' ? 'Try changing the filter' : 'System is running smoothly'}
            </p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <Alert key={alert.id} alert={alert} />
          ))
        )}
      </div>
    </div>
  )
}

export default LiveAlertsPanel
