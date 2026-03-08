import React, { useState, useEffect } from 'react'
import HealthCheckPanel from './components/HealthCheckPanel'
import ChatbotPanel from './components/ChatbotPanel'
import LiveAlertsPanel from './components/LiveAlertsPanel'
import { Activity, Moon, Sun, ShieldCheck } from 'lucide-react'

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const storedTheme = localStorage.getItem('theme')
    if (storedTheme) {
      return storedTheme === 'dark'
    }

    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  useEffect(() => {
    localStorage.setItem('theme', darkMode ? 'dark' : 'light')
  }, [darkMode])

  return (
    <div className={`min-h-screen antialiased ${darkMode ? 'dark' : ''}`}>
      {/* Header */}
      <header className="border-b border-blue-500/30 bg-gradient-to-r from-blue-700 via-blue-700 to-indigo-700 text-white shadow-xl">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between gap-4">
            <div className="flex items-center space-x-3">
              <div className="rounded-xl bg-white/15 p-2 backdrop-blur-sm">
                <Activity className="h-7 w-7" />
              </div>
              <div>
                <h1 className="text-2xl font-bold tracking-tight">AutoPilot AI</h1>
                <p className="text-sm font-medium text-blue-100/95">Multi-Agent SRE Control Center</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-3">
              <div className="hidden items-center gap-2 rounded-lg bg-white/10 px-3 py-1.5 text-xs text-blue-50 md:flex">
                <ShieldCheck className="h-4 w-4" />
                <span className="font-semibold">Environment:</span>
                <span>Development</span>
              </div>

              <button
                onClick={() => setDarkMode((prev) => !prev)}
                className="inline-flex items-center gap-2 rounded-lg border border-white/30 bg-white/10 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-white/20"
                title="Toggle theme"
              >
                {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
                <span>{darkMode ? 'Light' : 'Dark'} mode</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - 3 Panel Layout */}
      <div className="flex h-[calc(100vh-80px)] bg-slate-50 dark:bg-slate-950">
        {/* Left Panel - Health Checks */}
        <div className="w-80 overflow-y-auto border-r border-slate-200 bg-white/90 shadow-sm backdrop-blur dark:border-slate-800 dark:bg-slate-900/90 scrollbar-thin">
          <HealthCheckPanel />
        </div>

        {/* Middle Panel - Chatbot */}
        <div className="flex-1 overflow-hidden bg-slate-100/80 dark:bg-slate-950">
          <ChatbotPanel />
        </div>

        {/* Right Panel - Live Alerts */}
        <div className="w-96 overflow-y-auto border-l border-slate-200 bg-white/90 shadow-sm backdrop-blur dark:border-slate-800 dark:bg-slate-900/90 scrollbar-thin">
          <LiveAlertsPanel />
        </div>
      </div>
    </div>
  )
}

export default App
