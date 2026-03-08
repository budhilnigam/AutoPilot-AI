import React, { useState, useEffect } from 'react'
import HealthCheckPanel from './components/HealthCheckPanel'
import ChatbotPanel from './components/ChatbotPanel'
import LiveAlertsPanel from './components/LiveAlertsPanel'
import { Activity } from 'lucide-react'

function App() {
  const [darkMode, setDarkMode] = useState(false)

  return (
    <div className={`min-h-screen ${darkMode ? 'dark' : ''}`}>
      {/* Header */}
      <header className="bg-gradient-to-r from-blue-600 to-blue-800 text-white shadow-lg">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Activity className="w-8 h-8" />
              <div>
                <h1 className="text-2xl font-bold">AutoPilot AI</h1>
                <p className="text-blue-100 text-sm">Multi-Agent SRE System</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="text-sm text-blue-100">
                <span className="font-medium">Environment:</span> Development
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content - 3 Panel Layout */}
      <div className="flex h-[calc(100vh-80px)]">
        {/* Left Panel - Health Checks */}
        <div className="w-80 border-r border-gray-200 bg-white overflow-y-auto scrollbar-thin">
          <HealthCheckPanel />
        </div>

        {/* Middle Panel - Chatbot */}
        <div className="flex-1 bg-gray-50 overflow-hidden">
          <ChatbotPanel />
        </div>

        {/* Right Panel - Live Alerts */}
        <div className="w-96 border-l border-gray-200 bg-white overflow-y-auto scrollbar-thin">
          <LiveAlertsPanel />
        </div>
      </div>
    </div>
  )
}

export default App
