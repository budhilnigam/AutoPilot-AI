import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import { 
  Send, 
  Loader2, 
  Bot, 
  User,
  Sparkles,
  TrendingUp,
  AlertTriangle,
  Lightbulb
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const severityColors = {
  critical: 'bg-red-100 border-red-300 text-red-800',
  high: 'bg-orange-100 border-orange-300 text-orange-800',
  medium: 'bg-yellow-100 border-yellow-300 text-yellow-800',
  low: 'bg-blue-100 border-blue-300 text-blue-800',
  info: 'bg-gray-100 border-gray-300 text-gray-800',
}

function ChatbotPanel() {
  const [messages, setMessages] = useState([])
  const [inputMessage, setInputMessage] = useState('')
  const [loading, setLoading] = useState(false)
  const [defaultPrompts, setDefaultPrompts] = useState([])
  const messagesEndRef = useRef(null)

  useEffect(() => {
    // Fetch default prompts
    axios.get(`${API_BASE_URL}/api/prompts/default`)
      .then(res => setDefaultPrompts(res.data))
      .catch(err => console.error('Failed to load default prompts:', err))

    // Welcome message
    setMessages([{
      id: 1,
      type: 'bot',
      content: 'Welcome to AutoPilot AI! I\'m your SRE AI Copilot. I can help you analyze metrics, optimize costs, review infrastructure, and much more. What would you like to know?',
      timestamp: new Date(),
    }])
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const sendMessage = async (message) => {
    // Use optional chaining to safely check message, fallback to inputMessage
    const userMessage = message || inputMessage
    if (!userMessage || !userMessage.trim()) return

    const newUserMessage = {
      id: Date.now(),
      type: 'user',
      content: userMessage.trim(),
      timestamp: new Date(),
    }

    setMessages(prev => [...prev, newUserMessage])
    setInputMessage('')
    setLoading(true)

    try {
      const response = await axios.post(`${API_BASE_URL}/api/chat`, {
        message: userMessage,
      })

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.data.response,
        insights: response.data.insights,
        recommendations: response.data.recommendations,
        agentType: response.data.agent_type,
        executionTime: response.data.execution_time_ms,
        timestamp: new Date(),
      }

      setMessages(prev => [...prev, botMessage])
    } catch (error) {
      console.error('Chat error:', error)
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        error: true,
        timestamp: new Date(),
      }
      
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleSubmit = (e) => {
    e.preventDefault()
    sendMessage()
  }

  const handlePromptClick = (prompt) => {
    sendMessage(prompt.prompt)
  }

  const Message = ({ message }) => {
    const isBot = message.type === 'bot'

    return (
      <div className={`flex ${isBot ? 'justify-start' : 'justify-end'} mb-4`}>
        <div className={`flex items-start max-w-3xl ${isBot ? 'flex-row' : 'flex-row-reverse'}`}>
          {/* Avatar */}
          <div className={`flex-shrink-0 ${isBot ? 'mr-3' : 'ml-3'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${
              isBot ? 'bg-blue-600 text-white' : 'bg-gray-600 text-white'
            }`}>
              {isBot ? <Bot className="w-5 h-5" /> : <User className="w-5 h-5" />}
            </div>
          </div>

          {/* Message Content */}
          <div className={`flex-1 ${isBot ? '' : 'flex flex-col items-end'}`}>
            <div className={`px-4 py-3 rounded-lg ${
              isBot 
                ? message.error 
                  ? 'bg-red-50 text-red-900 border border-red-200' 
                  : 'bg-white border border-gray-200 shadow-sm' 
                : 'bg-blue-600 text-white'
            }`}>
              <p className="text-sm whitespace-pre-wrap">{message.content}</p>
              
              {/* Insights */}
              {message.insights && message.insights.length > 0 && (
                <div className="mt-3 space-y-2">
                  {message.insights.map((insight, idx) => (
                    <div 
                      key={idx} 
                      className={`p-3 rounded border ${severityColors[insight.severity] || severityColors.info}`}
                    >
                      <div className="flex items-start">
                        <AlertTriangle className="w-4 h-4 mr-2 mt-0.5 flex-shrink-0" />
                        <div className="flex-1">
                          <p className="font-medium text-sm">{insight.summary}</p>
                          <p className="text-xs mt-1 opacity-90">{insight.business_impact}</p>
                          {insight.confidence_score && (
                            <p className="text-xs mt-1 opacity-75">
                              Confidence: {(insight.confidence_score * 100).toFixed(0)}%
                            </p>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Recommendations */}
              {message.recommendations && message.recommendations.length > 0 && (
                <div className="mt-3 space-y-1">
                  <p className="text-xs font-semibold text-gray-700 flex items-center">
                    <Lightbulb className="w-3 h-3 mr-1" />
                    Recommendations:
                  </p>
                  {message.recommendations.map((rec, idx) => (
                    <div key={idx} className="flex items-start text-xs text-gray-700 ml-4">
                      <span className="mr-2">•</span>
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Metadata */}
              {isBot && message.agentType && (
                <div className="mt-2 pt-2 border-t border-gray-200 text-xs text-gray-500">
                  <span className="capitalize">{message.agentType}</span> Agent
                  {message.executionTime && (
                    <span className="ml-2">• {message.executionTime.toFixed(0)}ms</span>
                  )}
                </div>
              )}
            </div>

            <p className="text-xs text-gray-500 mt-1">
              {message.timestamp.toLocaleTimeString()}
            </p>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-full">
      {/* Default Prompts */}
      {messages.length <= 1 && defaultPrompts.length > 0 && (
        <div className="p-6 bg-gradient-to-b from-blue-50 to-transparent">
          <h3 className="text-sm font-semibold text-gray-700 mb-3 flex items-center">
            <Sparkles className="w-4 h-4 mr-2 text-blue-600" />
            Quick Actions
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {defaultPrompts.slice(0, 6).map((prompt) => (
              <button
                key={prompt.id}
                onClick={() => handlePromptClick(prompt)}
                className="p-3 text-left bg-white border border-gray-200 rounded-lg hover:border-blue-400 hover:shadow-md transition-all group"
              >
                <div className="flex items-start">
                  <span className="text-2xl mr-2">{prompt.icon}</span>
                  <div className="flex-1 min-w-0">
                    <h4 className="text-sm font-medium text-gray-900 group-hover:text-blue-600 truncate">
                      {prompt.title}
                    </h4>
                    <p className="text-xs text-gray-500 mt-1 line-clamp-2">
                      {prompt.prompt}
                    </p>
                  </div>
                </div>
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto scrollbar-thin p-6">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="flex items-center bg-white border border-gray-200 rounded-lg px-4 py-3 shadow-sm">
              <Loader2 className="w-5 h-5 text-blue-600 animate-spin mr-2" />
              <span className="text-sm text-gray-600">Analyzing...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 bg-white border-t border-gray-200">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask me anything about your infrastructure..."
            className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            disabled={loading}
          />
          <button
            type="submit"
            disabled={loading || !inputMessage.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  )
}

export default ChatbotPanel
