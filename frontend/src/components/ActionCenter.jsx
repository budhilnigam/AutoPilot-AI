import React, { useMemo, useState } from 'react'
import { CheckCircle2, ClipboardList, PlayCircle, ShieldCheck } from 'lucide-react'

const STAGES = ['Suggested', 'Approved', 'Executed', 'Verified']

function ActionCenter({ actions, onCreateAction, onUpdateAction }) {
  const [draft, setDraft] = useState('')

  const grouped = useMemo(() => {
    return STAGES.reduce((acc, stage) => {
      acc[stage] = actions.filter((item) => item.status === stage)
      return acc
    }, {})
  }, [actions])

  const submitDraft = () => {
    const value = draft.trim()
    if (!value) {
      return
    }

    onCreateAction(value)
    setDraft('')
  }

  const iconByStage = {
    Suggested: ClipboardList,
    Approved: ShieldCheck,
    Executed: PlayCircle,
    Verified: CheckCircle2,
  }

  return (
    <div className="space-y-4 p-4 md:p-6">
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Create Action</h3>
        <div className="mt-3 flex gap-2">
          <input
            type="text"
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            placeholder="Example: Reduce overprovisioned ECS task CPU by 20%"
            className="flex-1 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 focus:border-blue-500 focus:outline-none dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
          />
          <button onClick={submitDraft} className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700">Add</button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {STAGES.map((stage) => {
          const Icon = iconByStage[stage]
          return (
            <div key={stage} className="rounded-xl border border-slate-200 bg-white p-3 shadow-sm dark:border-slate-700 dark:bg-slate-900">
              <h4 className="mb-2 flex items-center text-xs font-semibold uppercase tracking-wide text-slate-600 dark:text-slate-300">
                <Icon className="mr-1 h-4 w-4" />
                {stage} ({grouped[stage].length})
              </h4>

              <div className="space-y-2">
                {grouped[stage].length === 0 ? (
                  <p className="text-xs text-slate-500 dark:text-slate-400">No actions</p>
                ) : (
                  grouped[stage].map((item) => (
                    <div key={item.id} className="rounded-lg border border-slate-200 bg-slate-50 p-2 dark:border-slate-700 dark:bg-slate-800/70">
                      <p className="text-xs text-slate-800 dark:text-slate-100">{item.title}</p>
                      <select
                        value={item.status}
                        onChange={(event) => onUpdateAction(item.id, event.target.value)}
                        className="mt-2 w-full rounded border border-slate-300 bg-white px-2 py-1 text-xs dark:border-slate-600 dark:bg-slate-900"
                      >
                        {STAGES.map((opt) => (
                          <option key={opt} value={opt}>{opt}</option>
                        ))}
                      </select>
                    </div>
                  ))
                )}
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default ActionCenter
