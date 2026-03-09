import React, { useState } from 'react'
import { Bookmark, Save } from 'lucide-react'

function PlaybooksPanel({ playbooks, onSavePlaybook, onRunPlaybook }) {
  const [name, setName] = useState('')
  const [prompt, setPrompt] = useState('')

  const savePlaybook = () => {
    const payload = {
      name: name.trim(),
      prompt: prompt.trim(),
    }

    if (!payload.name || !payload.prompt) {
      return
    }

    onSavePlaybook(payload)
    setName('')
    setPrompt('')
  }

  return (
    <div className="space-y-4 p-4 md:p-6">
      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="text-sm font-semibold text-slate-800 dark:text-slate-100">Save Playbook</h3>
        <div className="mt-3 grid grid-cols-1 gap-2">
          <input
            value={name}
            onChange={(event) => setName(event.target.value)}
            placeholder="Playbook name"
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
          />
          <textarea
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            rows={3}
            placeholder="Prompt template to run in chat"
            className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm dark:border-slate-700 dark:bg-slate-950"
          />
          <button onClick={savePlaybook} className="inline-flex items-center justify-center rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700">
            <Save className="mr-2 h-4 w-4" />
            Save Playbook
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm dark:border-slate-700 dark:bg-slate-900">
        <h3 className="mb-3 text-sm font-semibold text-slate-800 dark:text-slate-100">Saved Views / Playbooks</h3>
        <div className="space-y-2">
          {playbooks.length === 0 ? (
            <p className="text-xs text-slate-500 dark:text-slate-400">No playbooks yet.</p>
          ) : (
            playbooks.map((playbook) => (
              <div key={playbook.id} className="flex items-center justify-between rounded-lg border border-slate-200 bg-slate-50 p-3 dark:border-slate-700 dark:bg-slate-800/70">
                <div>
                  <p className="text-sm font-medium text-slate-800 dark:text-slate-100">{playbook.name}</p>
                  <p className="mt-1 text-xs text-slate-600 dark:text-slate-300 line-clamp-2">{playbook.prompt}</p>
                </div>
                <button
                  onClick={() => onRunPlaybook(playbook)}
                  className="ml-3 inline-flex items-center rounded-lg border border-blue-300 px-3 py-1 text-xs font-medium text-blue-700 hover:bg-blue-50 dark:border-blue-700 dark:text-blue-300 dark:hover:bg-blue-900/30"
                >
                  <Bookmark className="mr-1 h-3 w-3" />
                  Run
                </button>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}

export default PlaybooksPanel
