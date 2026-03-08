import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  RefreshCw,
  Activity,
  Server,
  Database,
  DollarSign,
  GitBranch,
  Layout,
  Cloud,
  Github
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const iconMap = {
  'Planner Agent': Layout,
  'Observability Agent': Activity,
  'Infrastructure Agent': Server,
  'Database Agent': Database,
  'Cost Agent': DollarSign,
  'CI/CD Agent': GitBranch,
  'Bedrock Client': Cloud,
  'Knowledge Base': Database,
  'CloudWatch': Activity,
  'GitHub': Github,
}

function HealthCheckPanel() {
  const [agentHealth, setAgentHealth] = useState([])
  const [serviceHealth, setServiceHealth] = useState([])
  const [loading, setLoading] = useState(true)
  const [lastUpdate, setLastUpdate] = useState(null)
  const [autoRefresh, setAutoRefresh] = useState(true)

  const fetchHealthData = async () => {
    try {
      setLoading(true)
      const [agentsRes, servicesRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/health/agents`),
        axios.get(`${API_BASE_URL}/api/health/services`)
      ])
      
      setAgentHealth(agentsRes.data)
      setServiceHealth(servicesRes.data)
      setLastUpdate(new Date())
    } catch (error) {
      console.error('Failed to fetch health data:', error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchHealthData()
    
    // Auto-refresh based on config (default: 5 minutes)
    if (autoRefresh) {
      const interval = setInterval(fetchHealthData, 5 * 60 * 1000)
      return () => clearInterval(interval)
    }
  }, [autoRefresh])

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy':
        return <CheckCircle2 className="w-5 h-5 text-green-500" />
      case 'degraded':
        return <AlertCircle className="w-5 h-5 text-yellow-500" />
      case 'unhealthy':
        return <XCircle className="w-5 h-5 text-red-500" />
      default:
        return <AlertCircle className="w-5 h-5 text-gray-400" />
    }
  }

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy':
        return 'border-green-200 bg-green-50 dark:border-green-900 dark:bg-green-950/40'
      case 'degraded':
        return 'border-yellow-200 bg-yellow-50 dark:border-yellow-900 dark:bg-yellow-950/40'
      case 'unhealthy':
        return 'border-red-200 bg-red-50 dark:border-red-900 dark:bg-red-950/40'
      default:
        return 'border-slate-200 bg-slate-50 dark:border-slate-700 dark:bg-slate-800/60'
    }
  }

  const HealthItem = ({ item }) => {
    const Icon = iconMap[item.service] || Activity
    const hasResponseTime = item.response_time_ms !== null && item.response_time_ms !== undefined

    const formatResponseTime = (value) => {
      const numericValue = Number(value)
      if (!Number.isFinite(numericValue)) {
        return null
      }

      // Health checks can complete in under 1ms; display that explicitly instead of "0".
      if (numericValue > 0 && numericValue < 1) {
        return '<1ms'
      }

      return `${Math.round(numericValue)}ms`
    }

    const responseTimeLabel = hasResponseTime ? formatResponseTime(item.response_time_ms) : null
    
    return (
      <div className={`rounded-lg border p-3 transition-all hover:shadow-md ${getStatusColor(item.status)}`}>
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-2 flex-1">
            <Icon className="mt-0.5 h-4 w-4 text-slate-600 dark:text-slate-300" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <h4 className="truncate text-sm font-medium text-slate-900 dark:text-slate-100">
                  {item.service}
                </h4>
                {getStatusIcon(item.status)}
              </div>
              <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">{item.message}</p>
              {responseTimeLabel && (
                <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
                  {responseTimeLabel}
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex h-full flex-col p-4">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-bold tracking-tight text-slate-900 dark:text-slate-100">System Health</h2>
          <button
            onClick={fetchHealthData}
            disabled={loading}
            className="rounded-lg p-2 transition-colors hover:bg-slate-100 disabled:opacity-50 dark:hover:bg-slate-800"
            title="Refresh"
          >
            <RefreshCw className={`h-4 w-4 text-slate-600 dark:text-slate-300 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
        
        {lastUpdate && (
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Last updated: {lastUpdate.toLocaleTimeString()}
          </p>
        )}
        
        <label className="flex items-center space-x-2 mt-2">
          <input
            type="checkbox"
            checked={autoRefresh}
            onChange={(e) => setAutoRefresh(e.target.checked)}
            className="rounded text-blue-600"
          />
          <span className="text-xs text-slate-600 dark:text-slate-300">Auto-refresh (5 min)</span>
        </label>
      </div>

      {/* Health Checks */}
      <div className="flex-1 overflow-y-auto scrollbar-thin space-y-6">
        {/* Agents Section */}
        <div>
          <h3 className="mb-3 flex items-center text-sm font-semibold text-slate-700 dark:text-slate-200">
            <Activity className="w-4 h-4 mr-2" />
            AI Agents ({agentHealth.length})
          </h3>
          <div className="space-y-2">
            {agentHealth.map((agent, idx) => (
              <HealthItem key={idx} item={agent} />
            ))}
          </div>
        </div>

        {/* Services Section */}
        <div>
          <h3 className="mb-3 flex items-center text-sm font-semibold text-slate-700 dark:text-slate-200">
            <Server className="w-4 h-4 mr-2" />
            Services ({serviceHealth.length})
          </h3>
          <div className="space-y-2">
            {serviceHealth.map((service, idx) => (
              <HealthItem key={idx} item={service} />
            ))}
          </div>
        </div>
      </div>

      {/* Summary Stats */}
      <div className="mt-4 border-t border-slate-200 pt-4 dark:border-slate-800">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-lg font-bold text-green-600">
              {[...agentHealth, ...serviceHealth].filter(h => h.status === 'healthy').length}
            </div>
            <div className="text-xs text-slate-600 dark:text-slate-300">Healthy</div>
          </div>
          <div>
            <div className="text-lg font-bold text-yellow-600">
              {[...agentHealth, ...serviceHealth].filter(h => h.status === 'degraded').length}
            </div>
            <div className="text-xs text-slate-600 dark:text-slate-300">Degraded</div>
          </div>
          <div>
            <div className="text-lg font-bold text-red-600">
              {[...agentHealth, ...serviceHealth].filter(h => h.status === 'unhealthy').length}
            </div>
            <div className="text-xs text-slate-600 dark:text-slate-300">Unhealthy</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HealthCheckPanel
