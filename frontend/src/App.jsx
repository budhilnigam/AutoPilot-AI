import React, { useEffect, useMemo, useState } from 'react'
import axios from 'axios'
import HealthCheckPanel from './components/HealthCheckPanel'
import ChatbotPanel from './components/ChatbotPanel'
import LiveAlertsPanel from './components/LiveAlertsPanel'
import CommandPalette from './components/CommandPalette'
import InsightsDashboard from './components/InsightsDashboard'
import ActionCenter from './components/ActionCenter'
import PlaybooksPanel from './components/PlaybooksPanel'
import ScenarioSimulatorPanel from './components/ScenarioSimulatorPanel'
import ExecutiveSnapshot from './components/ExecutiveSnapshot'
import AgentActivityFeed from './components/AgentActivityFeed'
import RepoTimeline from './components/RepoTimeline'
import AutonomousJobsPanel from './components/AutonomousJobsPanel'
import AuthPanel from './components/AuthPanel'
import AccountConnections from './components/AccountConnections'
import { Activity, Command, LogOut, Menu, Moon, ShieldCheck, Sun } from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const TABS = [
  { id: 'control', label: 'Control Center' },
  { id: 'insights', label: 'CI/CD Insights' },
  { id: 'actions', label: 'Action Center' },
  { id: 'playbooks', label: 'Playbooks' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'jobs', label: 'Autonomous Jobs' },
  { id: 'simulator', label: 'Simulator' },
  { id: 'executive', label: 'Executive' },
  { id: 'settings', label: 'Account Settings' },
]

const QUICK_ACTIONS = [
  {
    id: 'qa-cost',
    label: 'Run Cost Optimization Analysis',
    category: 'analysis',
    description: 'Analyze AWS costs and provide actionable savings opportunities.',
    prompt: 'Analyze current AWS costs and provide optimization recommendations in INR',
    targetTab: 'control',
  },
  {
    id: 'qa-failed-builds',
    label: 'Investigate Failed Builds',
    category: 'cicd',
    description: 'Diagnose recent workflow failures and suggest fixes.',
    prompt: 'Analyze recent CI/CD builds and identify any regressions or failures',
    targetTab: 'insights',
  },
  {
    id: 'qa-settings',
    label: 'Open Account Settings',
    category: 'navigation',
    description: 'Connect AWS and GitHub accounts.',
    targetTab: 'settings',
  },
]

const DEFAULT_EXECUTIVE_SUMMARY = {
  operationalRisk: 'Moderate',
  savingsOpportunity: '$0',
  deliveryHealth: 'Stable',
  securityPosture: 'Reviewing',
}

function App() {
  const [darkMode, setDarkMode] = useState(() => {
    const storedTheme = localStorage.getItem('theme')
    if (storedTheme) return storedTheme === 'dark'
    return window.matchMedia('(prefers-color-scheme: dark)').matches
  })

  const [authToken, setAuthToken] = useState(() => localStorage.getItem('autopilot-token') || '')
  const [currentUser, setCurrentUser] = useState(null)
  const [connections, setConnections] = useState(null)
  const [authError, setAuthError] = useState('')
  const [activeTab, setActiveTab] = useState('control')
  const [activeRole, setActiveRole] = useState('sre')
  const [commandPaletteOpen, setCommandPaletteOpen] = useState(false)
  const [queuedPrompt, setQueuedPrompt] = useState('')
  const [activities, setActivities] = useState([])
  const [actions, setActions] = useState([])
  const [jobs, setJobs] = useState([])
  const [playbooks, setPlaybooks] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem('autopilot-playbooks') || '[]')
    } catch {
      return []
    }
  })
  const [executiveSummary, setExecutiveSummary] = useState(DEFAULT_EXECUTIVE_SUMMARY)
  const [latestAwsContext, setLatestAwsContext] = useState(null)
  const [mobilePanelsOpen, setMobilePanelsOpen] = useState(false)
  const [isDesktop, setIsDesktop] = useState(() => window.innerWidth >= 1280)

  const awsConnected = Boolean(connections?.aws_connected)
  const githubConnected = Boolean(connections?.github_connected)

  useEffect(() => {
    localStorage.setItem('theme', darkMode ? 'dark' : 'light')
  }, [darkMode])

  useEffect(() => {
    localStorage.setItem('autopilot-playbooks', JSON.stringify(playbooks))
  }, [playbooks])

  useEffect(() => {
    if (authToken) {
      axios.defaults.headers.common.Authorization = `Bearer ${authToken}`
      localStorage.setItem('autopilot-token', authToken)
    } else {
      delete axios.defaults.headers.common.Authorization
      localStorage.removeItem('autopilot-token')
    }
  }, [authToken])

  const refreshSession = async (tokenOverride) => {
    const activeToken = tokenOverride || authToken
    if (!activeToken) {
      setCurrentUser(null)
      setConnections(null)
      return
    }
    try {
      const res = await axios.get(`${API_BASE_URL}/api/auth/me`, {
        headers: { Authorization: `Bearer ${activeToken}` },
      })
      setCurrentUser(res.data.user)
      setConnections(res.data.connections)
      setAuthError('')
    } catch (err) {
      setCurrentUser(null)
      setConnections(null)
      setAuthToken('')
      setAuthError(err?.response?.data?.detail || 'Session expired. Please login again.')
    }
  }

  useEffect(() => {
    refreshSession()
  }, [])

  useEffect(() => {
    const onResize = () => setIsDesktop(window.innerWidth >= 1280)
    window.addEventListener('resize', onResize)
    return () => window.removeEventListener('resize', onResize)
  }, [])

  useEffect(() => {
    const handleKeyDown = (event) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setCommandPaletteOpen(true)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  useEffect(() => {
    if (!authToken || !awsConnected) {
      setActions([])
      setJobs([])
      return
    }

    const loadOpsState = async () => {
      try {
        const [actionsRes, jobsRes, runsRes] = await Promise.all([
          axios.get(`${API_BASE_URL}/api/ops/actions`),
          axios.get(`${API_BASE_URL}/api/ops/jobs`),
          axios.get(`${API_BASE_URL}/api/ops/jobs/runs?limit=10`),
        ])

        setActions(actionsRes.data.actions || [])
        setJobs(jobsRes.data.jobs || [])

        const runActivities = (runsRes.data.runs || []).map((run) => ({
          id: `run-${run.job_id}-${run.completed_at}`,
          prompt: `Autonomous job: ${run.name}`,
          agentType: 'Scheduler',
          executionTime: 0,
          insightCount: run.insight_count || 0,
          recommendations: [],
          timestamp: run.completed_at,
        }))
        setActivities((prev) => [...runActivities, ...prev].slice(0, 20))
      } catch (error) {
        console.error('Failed to load ops state:', error)
      }
    }

    loadOpsState()
  }, [authToken, awsConnected])

  const rolePresets = useMemo(() => ({
    sre: ['control', 'insights', 'timeline', 'actions', 'jobs', 'settings'],
    finops: ['control', 'simulator', 'actions', 'executive', 'jobs', 'settings'],
    platform: ['control', 'insights', 'playbooks', 'timeline', 'actions', 'jobs', 'settings'],
  }), [])

  useEffect(() => {
    if (!rolePresets[activeRole]?.includes(activeTab)) {
      setActiveTab(rolePresets[activeRole][0])
    }
  }, [activeRole, activeTab, rolePresets])

  const visibleTabs = rolePresets[activeRole]
    .map((tabId) => TABS.find((tab) => tab.id === tabId))
    .filter(Boolean)

  const createAction = async (title) => {
    const cleaned = (title || '').trim()
    if (!cleaned || !awsConnected) return

    try {
      const response = await axios.post(`${API_BASE_URL}/api/ops/actions`, { title: cleaned })
      const created = response.data.action
      setActions((prev) => [created, ...prev.filter((item) => item.id !== created.id)])
    } catch (error) {
      console.error('Failed to create action:', error)
    }
  }

  const updateAction = async (id, status) => {
    try {
      const response = await axios.patch(`${API_BASE_URL}/api/ops/actions/${id}`, { status })
      const updated = response.data.action
      setActions((prev) => prev.map((action) => (action.id === id ? updated : action)))
    } catch (error) {
      console.error('Failed to update action:', error)
    }
  }

  const savePlaybook = ({ name, prompt }) => {
    setPlaybooks((prev) => [{ id: Date.now(), name, prompt }, ...prev])
  }

  const runPlaybook = (playbook) => {
    setActiveTab('control')
    setQueuedPrompt(playbook.prompt)
  }

  const createJob = async ({ name, prompt, schedule }) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/ops/jobs`, { name, prompt, schedule })
      const created = response.data.job
      setJobs((prev) => [created, ...prev.filter((item) => item.id !== created.id)])
    } catch (error) {
      console.error('Failed to create job:', error)
    }
  }

  const runJob = async (job) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/api/ops/jobs/${job.id}/run`)
      const run = response.data.run

      setJobs((prev) => prev.map((item) => {
        if (item.id !== job.id) return item
        return { ...item, last_run_at: run.completed_at, last_status: run.status }
      }))

      setActivities((prev) => [{
        id: `run-${run.job_id}-${run.completed_at}`,
        prompt: `Autonomous job executed: ${run.name}`,
        agentType: 'Scheduler',
        executionTime: 0,
        insightCount: run.insight_count || 0,
        recommendations: [],
        timestamp: run.completed_at,
      }, ...prev].slice(0, 20))
    } catch (error) {
      console.error('Failed to run autonomous job:', error)
    }
  }

  const toggleJob = async (id) => {
    const current = jobs.find((job) => job.id === id)
    if (!current) return

    try {
      const response = await axios.patch(`${API_BASE_URL}/api/ops/jobs/${id}`, { enabled: !current.enabled })
      const updated = response.data.job
      setJobs((prev) => prev.map((job) => (job.id === id ? updated : job)))
    } catch (error) {
      console.error('Failed to toggle job state:', error)
    }
  }

  const handleCommandExecute = (action) => {
    if (action.targetTab) setActiveTab(action.targetTab)
    if (action.prompt) setQueuedPrompt(action.prompt)
  }

  const handleChatActivity = (activity) => {
    setActivities((prev) => [activity, ...prev].slice(0, 20))

    if (activity.awsContext) setLatestAwsContext(activity.awsContext)

    if (activity.recommendations && activity.recommendations.length > 0) {
      const extracted = activity.recommendations.slice(0, 2)
      setExecutiveSummary((prev) => ({
        ...prev,
        deliveryHealth: activity.insightCount > 3 ? 'Attention Needed' : 'Stable',
        operationalRisk: activity.insightCount > 4 ? 'Elevated' : prev.operationalRisk,
        savingsOpportunity: activity.costHint || prev.savingsOpportunity,
      }))
      extracted.forEach((item) => createAction(item))
    }
  }

  const handleAuthenticated = (token, user) => {
    setAuthToken(token)
    setCurrentUser(user)
    refreshSession(token)
  }

  const logout = async () => {
    try {
      if (authToken) {
        await axios.post(`${API_BASE_URL}/api/auth/logout`, {}, { headers: { Authorization: `Bearer ${authToken}` } })
      }
    } catch (error) {
      console.error('Logout failed:', error)
    } finally {
      setAuthToken('')
      setCurrentUser(null)
      setConnections(null)
      setActiveTab('control')
    }
  }

  const renderMainContent = () => {
    if (activeTab === 'control') {
      return (
        <ChatbotPanel
          queuedPrompt={queuedPrompt}
          onPromptConsumed={() => setQueuedPrompt('')}
          onAgentActivity={handleChatActivity}
          awsConnected={awsConnected}
          awsMessage="AWS account connection is mandatory for core AI functionality."
        />
      )
    }
    if (activeTab === 'insights') {
      return (
        <InsightsDashboard
          githubConnected={githubConnected}
          githubMessage="GitHub is optional, but CI/CD insights stay disabled until a GitHub account is connected."
        />
      )
    }
    if (activeTab === 'actions') return <ActionCenter actions={actions} onCreateAction={createAction} onUpdateAction={updateAction} />
    if (activeTab === 'playbooks') return <PlaybooksPanel playbooks={playbooks} onSavePlaybook={savePlaybook} onRunPlaybook={runPlaybook} />
    if (activeTab === 'timeline') {
      return (
        <RepoTimeline
          activities={activities}
          githubConnected={githubConnected}
          githubMessage="GitHub timeline events are unavailable until you connect a GitHub account."
        />
      )
    }
    if (activeTab === 'jobs') return <AutonomousJobsPanel jobs={jobs} onCreateJob={createJob} onRunJob={runJob} onToggleJob={toggleJob} />
    if (activeTab === 'simulator') return <ScenarioSimulatorPanel />
    if (activeTab === 'executive') return <ExecutiveSnapshot summary={executiveSummary} />
    if (activeTab === 'settings') {
      return <div className="p-4 md:p-6"><AccountConnections token={authToken} connectionStatus={connections} onRefreshConnections={refreshSession} /></div>
    }
    return null
  }

  if (!authToken || !currentUser) {
    return <AuthPanel onAuthenticated={handleAuthenticated} />
  }

  return (
    <div className={`min-h-screen overflow-x-hidden antialiased ${darkMode ? 'dark' : ''}`}>
      <header className="border-b border-cyan-300/30 bg-[linear-gradient(120deg,#155e75,#0f172a_45%,#831843)] text-white shadow-xl">
        <div className="px-4 py-4 md:px-6">
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div className="rounded-xl bg-white/15 p-2 backdrop-blur-sm">
                <img src="/autopilot-ai-icon.png" alt="AutoPilot AI" className="h-6 w-6" />
              </div>
              <div>
                <h1 className="text-xl font-bold tracking-tight md:text-2xl">AutoPilot AI</h1>
                <p className="text-xs font-medium text-cyan-100 md:text-sm">Multi-Agent SRE Control Center</p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <div className="hidden items-center gap-2 rounded-lg bg-white/10 px-3 py-1.5 text-xs text-cyan-50 md:flex">
                <ShieldCheck className="h-4 w-4" />
                <span>{currentUser.email}</span>
              </div>

              <button
                onClick={() => setCommandPaletteOpen(true)}
                className="inline-flex items-center gap-2 rounded-lg border border-white/30 bg-white/10 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-white/20"
                title="Open command palette"
              >
                <Command className="h-4 w-4" />
                <span className="hidden md:inline">Commands</span>
              </button>

              <button
                onClick={() => setDarkMode((prev) => !prev)}
                className="inline-flex items-center gap-2 rounded-lg border border-white/30 bg-white/10 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-white/20"
                title="Toggle theme"
              >
                {darkMode ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
              </button>

              <button
                onClick={logout}
                className="inline-flex items-center gap-2 rounded-lg border border-white/30 bg-white/10 px-3 py-2 text-sm font-medium text-white transition-colors hover:bg-white/20"
                title="Logout"
              >
                <LogOut className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="border-b border-slate-200 bg-white/90 px-4 py-3 dark:border-slate-800 dark:bg-slate-900/90">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-wrap gap-2">
            {visibleTabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`rounded-lg px-3 py-1.5 text-xs font-semibold transition-all ${
                  activeTab === tab.id
                    ? 'bg-cyan-600 text-white shadow'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:hover:bg-slate-700'
                }`}
              >
                {tab.label}
              </button>
            ))}
          </div>

          <div className="flex items-center gap-2 text-xs">
            <span className="font-semibold text-slate-600 dark:text-slate-300">Role Workspace</span>
            <select
              value={activeRole}
              onChange={(event) => setActiveRole(event.target.value)}
              className="rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100"
            >
              <option value="sre">SRE</option>
              <option value="finops">FinOps</option>
              <option value="platform">Platform</option>
            </select>
            <button
              onClick={() => setMobilePanelsOpen((prev) => !prev)}
              className="inline-flex items-center gap-1 rounded-lg bg-slate-100 px-2 py-1 font-semibold dark:bg-slate-800 xl:hidden"
            >
              <Menu className="h-4 w-4" /> Panels
            </button>
          </div>
        </div>

        {!awsConnected && (
          <div className="mt-3 rounded-lg border border-amber-300 bg-amber-50 px-3 py-2 text-xs text-amber-800 dark:border-amber-900 dark:bg-amber-950/40 dark:text-amber-200">
            AWS account is not connected. Core functionality is disabled until you connect AWS in Account Settings.
          </div>
        )}

        {!githubConnected && (
          <div className="mt-2 rounded-lg border border-slate-300 bg-slate-50 px-3 py-2 text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
            GitHub is optional. CI/CD and repository features stay disabled until connected.
          </div>
        )}

        {authError && (
          <div className="mt-2 rounded-lg border border-rose-300 bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200">{authError}</div>
        )}

        {latestAwsContext && (
          <div className="mt-2 rounded-lg border border-cyan-200 bg-cyan-50 px-3 py-2 text-xs text-cyan-800 dark:border-cyan-900 dark:bg-cyan-950/30 dark:text-cyan-200">
            Account {latestAwsContext.account_id} in {latestAwsContext.region} | Monthly Unblended Cost: ${Number(latestAwsContext.monthly_unblended_cost_usd || 0).toFixed(2)} | EC2 CPU Metrics: {latestAwsContext.discovered_ec2_cpu_metrics}
          </div>
        )}
      </div>

      <div className="grid grid-cols-1 gap-3 bg-slate-100 p-3 dark:bg-slate-950 xl:grid-cols-[18rem_minmax(0,1fr)_22rem] xl:gap-4 xl:p-4 xl:min-h-[calc(100vh-12.2rem)]">
        {(mobilePanelsOpen || isDesktop) && (
          <aside className="order-1 max-h-[60vh] overflow-y-auto rounded-2xl border border-slate-200 bg-white/90 shadow-sm backdrop-blur dark:border-slate-800 dark:bg-slate-900/90 xl:order-none xl:max-h-none scrollbar-thin">
            <HealthCheckPanel />
          </aside>
        )}

        <main className="order-2 min-w-0 overflow-hidden rounded-2xl border border-slate-200 bg-white/75 shadow-sm dark:border-slate-800 dark:bg-slate-900/70 xl:order-none">
          {renderMainContent()}
        </main>

        {(mobilePanelsOpen || isDesktop) && (
          <aside className="order-3 max-h-[60vh] overflow-y-auto rounded-2xl border border-slate-200 bg-white/90 p-3 shadow-sm backdrop-blur dark:border-slate-800 dark:bg-slate-900/90 xl:order-none xl:max-h-none scrollbar-thin">
            <div className="space-y-3">
              <LiveAlertsPanel />
              <AgentActivityFeed activities={activities} />
            </div>
          </aside>
        )}
      </div>

      <CommandPalette
        open={commandPaletteOpen}
        onClose={() => setCommandPaletteOpen(false)}
        actions={QUICK_ACTIONS}
        onExecute={handleCommandExecute}
      />
    </div>
  )
}

export default App
