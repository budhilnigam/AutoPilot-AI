import React, { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import { GitCommitHorizontal, Workflow, Wrench } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function RepoTimeline({ activities, githubConnected = false, githubMessage = '' }) {
  const [commits, setCommits] = useState([])
  const [runs, setRuns] = useState([])

  useEffect(() => {
    if (!githubConnected) {
      setCommits([])
      setRuns([])
      return
    }

    const fetchData = async () => {
      try {
        const [commitRes, runRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/api/github/commits?limit=8`),
          axios.get(`${API_BASE_URL}/api/github/workflows?limit=8`),
        ])
        setCommits(commitRes.data.commits || [])
        setRuns(runRes.data.runs || [])
      } catch (error) {
        console.error('Failed to fetch repository timeline:', error)
      }
    }

    fetchData()
  }, [githubConnected])

  const mergedTimeline = useMemo(() => {
    const commitItems = commits.map((commit) => ({
      id: `c-${commit.sha}`,
      type: 'commit',
      title: commit.message,
      subtitle: `${commit.author} | ${commit.sha}`,
      timestamp: commit.date,
    }))

    const runItems = runs.map((run) => ({
      id: `r-${run.id}`,
      type: 'workflow',
      title: `${run.name} (${run.conclusion || run.status})`,
      subtitle: `${run.branch || 'unknown branch'} | #${run.run_number || 'n/a'}`,
      timestamp: run.updated_at || run.created_at,
    }))

    const activityItems = activities.slice(0, 8).map((item) => ({
      id: `a-${item.id}`,
      type: 'agent',
      title: `${item.agentType || 'Unknown'} agent analyzed request`,
      subtitle: item.prompt,
      timestamp: item.timestamp,
    }))

    return [...commitItems, ...runItems, ...activityItems]
      .filter((item) => item.timestamp)
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
      .slice(0, 20)
  }, [activities, commits, runs])

  const iconFor = (type) => {
    if (type === 'commit') return GitCommitHorizontal
    if (type === 'workflow') return Workflow
    return Wrench
  }

  return (
    <div className="space-y-4 p-4 md:p-6">
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Repo Intelligence Timeline</h3>
        <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">Commits + workflow runs + AI agent analyses in one feed.</p>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="space-y-3">
          {mergedTimeline.length === 0 ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              {githubConnected
                ? 'Timeline data unavailable. Configure GitHub repository details or trigger agent activity.'
                : githubMessage || 'GitHub is not connected. Connect it in Account Settings to unlock CI/CD timeline events.'}
            </p>
          ) : (
            mergedTimeline.map((event) => {
              const Icon = iconFor(event.type)
              return (
                <div key={event.id} className="flex items-start gap-3 rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800/70">
                  <div className="rounded-full bg-blue-100 p-2 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="min-w-0 flex-1">
                    <p className="text-sm font-medium text-slate-800 dark:text-slate-100">{event.title}</p>
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-300 line-clamp-2">{event.subtitle}</p>
                    <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">{formatDistanceToNow(new Date(event.timestamp), { addSuffix: true })}</p>
                  </div>
                </div>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}

export default RepoTimeline
