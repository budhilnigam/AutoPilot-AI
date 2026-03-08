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
        return 'border-green-200 bg-green-50'
      case 'degraded':
        return 'border-yellow-200 bg-yellow-50'
      case 'unhealthy':
        return 'border-red-200 bg-red-50'
      default:
        return 'border-gray-200 bg-gray-50'
    }
  }

  const HealthItem = ({ item }) => {
    const Icon = iconMap[item.service] || Activity
    
    return (
      <div className={`p-3 border rounded-lg ${getStatusColor(item.status)} transition-all hover:shadow-md`}>
        <div className="flex items-start justify-between">
          <div className="flex items-start space-x-2 flex-1">
            <Icon className="w-4 h-4 mt-0.5 text-gray-600" />
            <div className="flex-1 min-w-0">
              <div className="flex items-center justify-between">
                <h4 className="text-sm font-medium text-gray-900 truncate">
                  {item.service}
                </h4>
                {getStatusIcon(item.status)}
              </div>
              <p className="text-xs text-gray-600 mt-1">{item.message}</p>
              {item.response_time_ms && (
                <p className="text-xs text-gray-500 mt-1">
                  {item.response_time_ms.toFixed(0)}ms
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="p-4 h-full flex flex-col">
      {/* Header */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-lg font-bold text-gray-900">System Health</h2>
          <button
            onClick={fetchHealthData}
            disabled={loading}
            className="p-2 hover:bg-gray-100 rounded-lg transition-colors disabled:opacity-50"
            title="Refresh"
          >
            <RefreshCw className={`w-4 h-4 text-gray-600 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
        
        {lastUpdate && (
          <p className="text-xs text-gray-500">
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
          <span className="text-xs text-gray-600">Auto-refresh (5 min)</span>
        </label>
      </div>

      {/* Health Checks */}
      <div className="flex-1 overflow-y-auto scrollbar-thin space-y-6">
        {/* Agents Section */}
        <div>
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
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
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
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
      <div className="mt-4 pt-4 border-t border-gray-200">
        <div className="grid grid-cols-3 gap-2 text-center">
          <div>
            <div className="text-lg font-bold text-green-600">
              {[...agentHealth, ...serviceHealth].filter(h => h.status === 'healthy').length}
            </div>
            <div className="text-xs text-gray-600">Healthy</div>
          </div>
          <div>
            <div className="text-lg font-bold text-yellow-600">
              {[...agentHealth, ...serviceHealth].filter(h => h.status === 'degraded').length}
            </div>
            <div className="text-xs text-gray-600">Degraded</div>
          </div>
          <div>
            <div className="text-lg font-bold text-red-600">
              {[...agentHealth, ...serviceHealth].filter(h => h.status === 'unhealthy').length}
            </div>
            <div className="text-xs text-gray-600">Unhealthy</div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default HealthCheckPanel
