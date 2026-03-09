import React, { useEffect, useMemo, useState } from 'react'
import { Search, Zap } from 'lucide-react'

function CommandPalette({ open, onClose, actions, onExecute }) {
  const [query, setQuery] = useState('')

  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    if (open) {
      window.addEventListener('keydown', handleKeyDown)
    }

    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [open, onClose])

  useEffect(() => {
    if (!open) {
      setQuery('')
    }
  }, [open])

  const filteredActions = useMemo(() => {
    const normalized = query.trim().toLowerCase()
    if (!normalized) {
      return actions
    }

    return actions.filter((action) => {
      return (
        action.label.toLowerCase().includes(normalized)
        || action.description.toLowerCase().includes(normalized)
        || action.category.toLowerCase().includes(normalized)
      )
    })
  }, [actions, query])

  if (!open) {
    return null
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center bg-slate-900/60 p-4 pt-20 backdrop-blur-sm" onClick={onClose}>
      <div className="w-full max-w-2xl rounded-xl border border-slate-200 bg-white shadow-2xl dark:border-slate-700 dark:bg-slate-900" onClick={(event) => event.stopPropagation()}>
        <div className="flex items-center border-b border-slate-200 px-4 py-3 dark:border-slate-700">
          <Search className="mr-2 h-4 w-4 text-slate-500 dark:text-slate-400" />
          <input
            autoFocus
            type="text"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Run action, open tab, or trigger a workflow..."
            className="w-full bg-transparent text-sm text-slate-900 outline-none placeholder:text-slate-500 dark:text-slate-100 dark:placeholder:text-slate-400"
          />
          <span className="rounded border border-slate-300 px-2 py-0.5 text-xs text-slate-500 dark:border-slate-600 dark:text-slate-400">ESC</span>
        </div>

        <div className="max-h-96 overflow-y-auto py-2 scrollbar-thin">
          {filteredActions.length === 0 ? (
            <p className="px-4 py-8 text-center text-sm text-slate-500 dark:text-slate-400">No matching commands found.</p>
          ) : (
            filteredActions.map((action) => (
              <button
                key={action.id}
                onClick={() => {
                  onExecute(action)
                  onClose()
                }}
                className="flex w-full items-start justify-between px-4 py-3 text-left transition-colors hover:bg-slate-100 dark:hover:bg-slate-800"
              >
                <div>
                  <p className="text-sm font-medium text-slate-900 dark:text-slate-100">{action.label}</p>
                  <p className="mt-1 text-xs text-slate-600 dark:text-slate-300">{action.description}</p>
                </div>
                <div className="ml-2 flex items-center rounded-full bg-blue-100 px-2 py-1 text-[10px] font-semibold uppercase text-blue-700 dark:bg-blue-900/40 dark:text-blue-300">
                  <Zap className="mr-1 h-3 w-3" />
                  {action.category}
                </div>
              </button>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default CommandPalette
