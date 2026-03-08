import React, { useState, useEffect, useRef } from 'react'
import axios from 'axios'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeRaw from 'rehype-raw'
import rehypeSanitize from 'rehype-sanitize'
import { 
  Send, 
  Loader2, 
  Bot, 
  User,
  Sparkles,
  TrendingUp,
  AlertTriangle,
  Lightbulb,
  Brain
} from 'lucide-react'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const severityColors = {
  critical: 'bg-red-100 border-red-300 text-red-800 dark:bg-red-950/50 dark:border-red-800 dark:text-red-200',
  high: 'bg-orange-100 border-orange-300 text-orange-800 dark:bg-orange-950/50 dark:border-orange-800 dark:text-orange-200',
  medium: 'bg-yellow-100 border-yellow-300 text-yellow-800 dark:bg-yellow-950/50 dark:border-yellow-800 dark:text-yellow-200',
  low: 'bg-blue-100 border-blue-300 text-blue-800 dark:bg-blue-950/50 dark:border-blue-800 dark:text-blue-200',
  info: 'bg-gray-100 border-gray-300 text-gray-800 dark:bg-slate-800 dark:border-slate-700 dark:text-slate-200',
}

// Markdown Renderer Component
const MarkdownRenderer = ({ content, className = '' }) => {
  return (
    <div className={`markdown-content ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeRaw, rehypeSanitize]}
        components={{
          // Custom styling for markdown elements
          h1: ({node, ...props}) => <h1 className="text-2xl font-bold mt-4 mb-2" {...props} />,
          h2: ({node, ...props}) => <h2 className="text-xl font-bold mt-3 mb-2" {...props} />,
          h3: ({node, ...props}) => <h3 className="text-lg font-semibold mt-2 mb-1" {...props} />,
          p: ({node, ...props}) => <p className="mb-2 last:mb-0" {...props} />,
          ul: ({node, ...props}) => <ul className="list-disc list-inside mb-2 space-y-1" {...props} />,
          ol: ({node, ...props}) => <ol className="list-decimal list-inside mb-2 space-y-1" {...props} />,
          li: ({node, ...props}) => <li className="ml-2" {...props} />,
          code: ({node, inline, ...props}) => 
            inline 
              ? <code className="rounded bg-slate-100 px-1.5 py-0.5 text-sm font-mono dark:bg-slate-700" {...props} />
              : <code className="my-2 block overflow-x-auto rounded-lg bg-slate-900 p-3 font-mono text-sm text-slate-100" {...props} />,
          pre: ({node, ...props}) => <pre className="my-2" {...props} />,
          blockquote: ({node, ...props}) => <blockquote className="my-2 border-l-4 border-slate-300 pl-4 italic dark:border-slate-600" {...props} />,
          a: ({node, ...props}) => <a className="text-blue-600 underline hover:text-blue-800 dark:text-blue-300 dark:hover:text-blue-200" target="_blank" rel="noopener noreferrer" {...props} />,
          table: ({node, ...props}) => <div className="my-2 overflow-x-auto"><table className="min-w-full border border-slate-300 dark:border-slate-600" {...props} /></div>,
          th: ({node, ...props}) => <th className="border border-slate-300 bg-slate-100 px-3 py-2 text-left font-semibold dark:border-slate-600 dark:bg-slate-800" {...props} />,
          td: ({node, ...props}) => <td className="border border-slate-300 px-3 py-2 dark:border-slate-600" {...props} />,
          hr: ({node, ...props}) => <hr className="my-4 border-slate-300 dark:border-slate-600" {...props} />,
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
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

  // Extract reasoning from content
  const extractReasoning = (content) => {
    const reasoningRegex = /<reasoning>(.*?)<\/reasoning>/is
    const match = content.match(reasoningRegex)
    
    if (match) {
      const reasoning = match[1].trim()
      const cleanedContent = content.replace(reasoningRegex, '').trim()
      return { reasoning, cleanedContent }
    }
    
    return { reasoning: null, cleanedContent: content }
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

      // Extract reasoning from response content
      const { reasoning, cleanedContent } = extractReasoning(response.data.response)

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: cleanedContent,
        reasoning: reasoning,
        thinking: response.data.thinking,
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
              isBot ? 'bg-blue-600 text-white' : 'bg-slate-600 text-white dark:bg-slate-700'
            }`}>
              {isBot ? <Bot className="w-5 h-5" /> : <User className="w-5 h-5" />}
            </div>
          </div>

          {/* Message Content */}
          <div className={`flex-1 ${isBot ? '' : 'flex flex-col items-end'}`}>
            <div className={`px-4 py-3 rounded-lg ${
              isBot 
                ? message.error 
                  ? 'border border-red-200 bg-red-50 text-red-900 dark:border-red-800 dark:bg-red-950/50 dark:text-red-200' 
                  : 'border border-slate-200 bg-white shadow-sm dark:border-slate-700 dark:bg-slate-900' 
                : 'bg-blue-600 text-white'
            }`}>
              {/* Reasoning Process (Extracted from content) */}
              {message.reasoning && (
                <details className="mb-3 rounded border border-purple-300 bg-purple-50">
                  <summary className="flex cursor-pointer select-none items-center px-3 py-2 text-xs font-semibold text-purple-700 hover:bg-purple-100 dark:border-purple-700 dark:bg-purple-950/40 dark:text-purple-200 dark:hover:bg-purple-900/50">
                    <Brain className="w-4 h-4 mr-2" />
                    Reasoning
                  </summary>
                  <div className="border-t border-purple-200 bg-white px-3 py-2 dark:border-purple-700 dark:bg-slate-900">
                    <div className="text-xs text-slate-700 dark:text-slate-200">
                      <MarkdownRenderer content={message.reasoning} className="text-slate-700 dark:text-slate-200" />
                    </div>
                  </div>
                </details>
              )}

              {/* Main Response */}
              <div className="text-sm">
                <MarkdownRenderer 
                  content={message.content} 
                  className={isBot ? 'text-slate-900 dark:text-slate-100' : 'text-white markdown-white'} 
                />
              </div>

              {/* Thinking Process (Separate from Response) */}
              {message.thinking && (
                <details className="mt-3 rounded border border-slate-300 bg-slate-50 dark:border-slate-700 dark:bg-slate-800">
                  <summary className="flex cursor-pointer select-none items-center px-3 py-2 text-xs font-semibold text-slate-700 hover:bg-slate-100 dark:text-slate-200 dark:hover:bg-slate-700">
                    <Brain className="w-4 h-4 mr-2" />
                    Thinking Process
                  </summary>
                  <div className="border-t border-slate-200 bg-white px-3 py-2 dark:border-slate-700 dark:bg-slate-900">
                    <div className="text-xs text-slate-700 dark:text-slate-200">
                      <MarkdownRenderer content={message.thinking} className="text-slate-700 dark:text-slate-200" />
                    </div>
                  </div>
                </details>
              )}
              
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
                  <p className="flex items-center text-xs font-semibold text-slate-700 dark:text-slate-300">
                    <Lightbulb className="w-3 h-3 mr-1" />
                    Recommendations:
                  </p>
                  {message.recommendations.map((rec, idx) => (
                    <div key={idx} className="ml-4 flex items-start text-xs text-slate-700 dark:text-slate-300">
                      <span className="mr-2">•</span>
                      <span>{rec}</span>
                    </div>
                  ))}
                </div>
              )}

              {/* Metadata */}
              {isBot && message.agentType && (
                <div className="mt-2 border-t border-slate-200 pt-2 text-xs text-slate-500 dark:border-slate-700 dark:text-slate-400">
                  <span className="capitalize">{message.agentType}</span> Agent
                  {message.executionTime && (
                    <span className="ml-2">• {message.executionTime.toFixed(0)}ms</span>
                  )}
                </div>
              )}
            </div>

            <p className="mt-1 text-xs text-slate-500 dark:text-slate-400">
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
        <div className="bg-gradient-to-b from-blue-50 to-transparent p-6 dark:from-slate-900 dark:to-transparent">
          <h3 className="mb-3 flex items-center text-sm font-semibold text-slate-700 dark:text-slate-200">
            <Sparkles className="w-4 h-4 mr-2 text-blue-600" />
            Quick Actions
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {defaultPrompts.slice(0, 6).map((prompt) => (
              <button
                key={prompt.id}
                onClick={() => handlePromptClick(prompt)}
                className="group rounded-lg border border-slate-200 bg-white p-3 text-left transition-all hover:border-blue-400 hover:shadow-md dark:border-slate-700 dark:bg-slate-900 dark:hover:border-blue-500"
              >
                <div className="flex items-start">
                  <span className="text-2xl mr-2">{prompt.icon}</span>
                  <div className="flex-1 min-w-0">
                    <h4 className="truncate text-sm font-medium text-slate-900 group-hover:text-blue-600 dark:text-slate-100 dark:group-hover:text-blue-300">
                      {prompt.title}
                    </h4>
                    <p className="mt-1 line-clamp-2 text-xs text-slate-500 dark:text-slate-400">
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
      <div className="flex-1 overflow-y-auto p-6 scrollbar-thin">
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="flex items-center rounded-lg border border-slate-200 bg-white px-4 py-3 shadow-sm dark:border-slate-700 dark:bg-slate-900">
              <Loader2 className="w-5 h-5 text-blue-600 animate-spin mr-2" />
              <span className="text-sm text-slate-600 dark:text-slate-300">Analyzing...</span>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-slate-200 bg-white p-4 dark:border-slate-800 dark:bg-slate-900/90">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="Ask me anything about your infrastructure..."
            className="flex-1 rounded-lg border border-slate-300 bg-white px-4 py-3 text-slate-900 placeholder:text-slate-500 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-blue-500 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-400"
          />
          <button
            type="submit"
            disabled={loading || !inputMessage.trim()}
            className="flex items-center rounded-lg bg-blue-600 px-6 py-3 text-white transition-colors hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
      </div>
    </div>
  )
}

export default ChatbotPanel
