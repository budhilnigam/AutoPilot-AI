import React, { useEffect, useState } from 'react'
import axios from 'axios'
import { AlertCircle, CheckCircle2, GitBranch, TrendingUp } from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function MetricCard({ title, value, hint, icon: Icon, tone = 'blue' }) {
  const toneClasses = {
    blue: 'text-blue-700 bg-blue-50 border-blue-200 dark:text-blue-300 dark:bg-blue-950/30 dark:border-blue-900',
    green: 'text-green-700 bg-green-50 border-green-200 dark:text-green-300 dark:bg-green-950/30 dark:border-green-900',
    amber: 'text-amber-700 bg-amber-50 border-amber-200 dark:text-amber-300 dark:bg-amber-950/30 dark:border-amber-900',
    rose: 'text-rose-700 bg-rose-50 border-rose-200 dark:text-rose-300 dark:bg-rose-950/30 dark:border-rose-900',
  }

  return (
    <div className={`rounded-xl border p-4 ${toneClasses[tone]}`}>
      <div className="flex items-center justify-between">
        <p className="text-xs font-semibold uppercase tracking-wide">{title}</p>
        <Icon className="h-4 w-4" />
      </div>
      <p className="mt-2 text-2xl font-bold">{value}</p>
      <p className="mt-1 text-xs opacity-80">{hint}</p>
    </div>
  )
}

function MiniBars({ values }) {
  const max = Math.max(...values, 1)
  return (
    <div className="mt-2 flex h-16 items-end gap-1">
      {values.map((value, index) => (
        <div
          key={`${value}-${index}`}
          className="flex-1 rounded-t bg-blue-500/75 dark:bg-blue-400/70"
          style={{ height: `${Math.max((value / max) * 100, 8)}%` }}
          title={`${value}`}
        />
      ))}
    </div>
  )
}

function InsightsDashboard({ githubConnected = false, githubMessage = '' }) {
  const [loading, setLoading] = useState(true)
  const [health, setHealth] = useState(null)
  const [trends, setTrends] = useState(null)
  const [failedBuilds, setFailedBuilds] = useState([])

  useEffect(() => {
    if (!githubConnected) {
      setLoading(false)
      setHealth(null)
      return
    }

    const fetchData = async () => {
      try {
        setLoading(true)
        const [healthRes, trendRes, failedRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/api/github/builds/health`),
          axios.get(`${API_BASE_URL}/api/github/builds/trends?days=14`),
          axios.get(`${API_BASE_URL}/api/github/builds/failed?limit=5`),
        ])

        setHealth(healthRes.data)
        setTrends(trendRes.data)
        setFailedBuilds(failedRes.data.builds || [])
      } catch (error) {
        console.error('Failed to fetch CI/CD dashboard data:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [githubConnected])

  if (loading) {
    return <div className="p-6 text-sm text-slate-600 dark:text-slate-300">Loading CI/CD insights...</div>
  }

  if (!health) {
    return (
      <div className="p-6 text-sm text-slate-600 dark:text-slate-300">
        {githubConnected
          ? 'CI/CD data unavailable. Configure repository details or try again later.'
          : githubMessage || 'GitHub is not connected. Connect it in Account Settings to unlock CI/CD insights.'}
      </div>
    )
  }

  const successRate = Number(health.success_rate || 0)
  const avgBuildTime = Number(health.average_build_time_seconds || 0)
  const totalBuilds = Number(health.total_builds || 0)
  const failedCount = Number(health.failed_builds || 0)

  const trendValues = (trends?.daily_stats || []).map((item) => Number(item.total_builds || 0))

  return (
    <div className="space-y-4 p-4 md:p-6">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-4">
        <MetricCard title="Success Rate" value={`${successRate.toFixed(1)}%`} hint="Last analyzed window" icon={CheckCircle2} tone="green" />
        <MetricCard title="Avg Build Time" value={`${Math.round(avgBuildTime)}s`} hint="Across workflow runs" icon={TrendingUp} tone="blue" />
        <MetricCard title="Total Builds" value={totalBuilds} hint="Recent workflow runs" icon={GitBranch} tone="amber" />
        <MetricCard title="Failed Builds" value={failedCount} hint="Needs investigation" icon={AlertCircle} tone="rose" />
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Build Volume Trend (14 days)</h3>
        {trendValues.length > 0 ? (
          <MiniBars values={trendValues} />
        ) : (
          <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">No trend data available.</p>
        )}
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-100">Recent Failed Builds</h3>
        <div className="space-y-2">
          {failedBuilds.length === 0 ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">No failed builds in the selected period.</p>
          ) : (
            failedBuilds.map((build) => (
              <div key={build.build_id} className="rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800/70">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-medium text-slate-800 dark:text-slate-100">Build #{build.build_id}</p>
                  <span className="text-xs text-slate-500 dark:text-slate-400">{build.branch}</span>
                </div>
                <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">Commit: {build.commit_sha}</p>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default InsightsDashboard
