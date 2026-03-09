import React, { useState } from 'react'
import { CalendarClock, Play, Plus } from 'lucide-react'

function AutonomousJobsPanel({ jobs, onCreateJob, onRunJob, onToggleJob }) {
  const [name, setName] = useState('')
  const [prompt, setPrompt] = useState('')
  const [schedule, setSchedule] = useState('daily')

  const addJob = () => {
    const payload = {
      name: name.trim(),
      prompt: prompt.trim(),
      schedule,
    }

    if (!payload.name || !payload.prompt) {
      return
    }

    onCreateJob(payload)
    setName('')
    setPrompt('')
    setSchedule('daily')
  }

  return (
    <div className="space-y-4 p-4 md:p-6">
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="flex items-center text-sm font-semibold text-slate-800 dark:text-slate-100">
          <CalendarClock className="mr-2 h-4 w-4 text-blue-600" />
          Autonomous Jobs
        </h3>
        <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">Create scheduled analysis jobs and run them on demand from the control center.</p>

        <div className="mt-3 grid grid-cols-1 gap-2 md:grid-cols-4">
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Job name"
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
          />
          <input
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            placeholder="Prompt to execute"
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm md:col-span-2 dark:border-slate-700 dark:bg-slate-950"
          />
          <select
            value={schedule}
            onChange={(event) => setSchedule(event.target.value)}
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
          >
            <option value="hourly">Hourly</option>
            <option value="daily">Daily</option>
            <option value="weekly">Weekly</option>
          </select>
        </div>

        <button onClick={addJob} className="mt-3 inline-flex items-center rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700">
          <Plus className="mr-2 h-4 w-4" />
          Create Job
        </button>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <div className="space-y-2">
          {jobs.length === 0 ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">No jobs configured yet.</p>
          ) : (
            jobs.map((job) => (
              <div key={job.id} className="rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800/70">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div>
                    <p className="text-sm font-medium text-slate-800 dark:text-slate-100">{job.name}</p>
                    <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">{job.schedule} | {job.prompt}</p>
                  </div>
                  <div className="flex gap-2">
                    <button
                      onClick={() => onRunJob(job)}
                      className="inline-flex items-center rounded-lg border border-blue-300 px-3 py-1 text-xs font-medium text-blue-700 hover:bg-blue-50 dark:border-blue-700 dark:text-blue-300 dark:hover:bg-blue-900/30"
                    >
                      <Play className="mr-1 h-3 w-3" />
                      Run Now
                    </button>
                    <button
                      onClick={() => onToggleJob(job.id)}
                      className={`rounded-lg px-3 py-1 text-xs font-medium ${job.enabled ? 'bg-green-600 text-white' : 'bg-slate-200 text-slate-700 dark:bg-slate-700 dark:text-slate-200'}`}
                    >
                      {job.enabled ? 'Enabled' : 'Disabled'}
                    </button>
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default AutonomousJobsPanel
