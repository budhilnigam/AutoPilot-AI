import React from 'react'
import { Brain, Clock3, Cpu } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

function AgentActivityFeed({ activities }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
      <h3 className="mb-3 flex items-center text-sm font-semibold text-slate-800 dark:text-slate-100">
        <Brain className="mr-2 h-4 w-4 text-blue-600" />
        Agent Activity
      </h3>

      <div className="space-y-3">
        {activities.length === 0 ? (
          <p className="text-xs text-slate-500 dark:text-slate-400">No agent activity yet. Ask a question in chat to start tracking.</p>
        ) : (
          activities.map((activity) => (
            <div key={activity.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800/70">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wide text-blue-700 dark:text-blue-300">{activity.agentType || 'Unknown'} Agent</span>
                <span className="text-xs text-slate-500 dark:text-slate-400">{formatDistanceToNow(new Date(activity.timestamp), { addSuffix: true })}</span>
              </div>
              <p className="mt-2 text-xs text-slate-700 dark:text-slate-200 line-clamp-2">{activity.prompt}</p>
              <div className="mt-2 flex items-center gap-3 text-xs text-slate-500 dark:text-slate-400">
                <span className="inline-flex items-center"><Clock3 className="mr-1 h-3 w-3" />{Math.round(activity.executionTime || 0)}ms</span>
                <span className="inline-flex items-center"><Cpu className="mr-1 h-3 w-3" />{activity.insightCount || 0} insights</span>
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

export default AgentActivityFeed
