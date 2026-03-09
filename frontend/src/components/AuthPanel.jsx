import React, { useState } from 'react'
import axios from 'axios'
import { LockKeyhole, Mail, Sparkles } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Input } from './ui/input'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function AuthPanel({ onAuthenticated }) {
  const [mode, setMode] = useState('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  const submit = async (event) => {
    event.preventDefault()
    setLoading(true)
    setError('')

    try {
      const endpoint = mode === 'login' ? '/api/auth/login' : '/api/auth/register'
      const res = await axios.post(`${API_BASE_URL}${endpoint}`, { email, password })
      onAuthenticated(res.data.token, res.data.user)
    } catch (err) {
      setError(err?.response?.data?.detail || 'Authentication failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[radial-gradient(circle_at_20%_20%,#67e8f9_0%,transparent_45%),radial-gradient(circle_at_80%_10%,#fda4af_0%,transparent_40%),linear-gradient(135deg,#f8fafc_0%,#e2e8f0_55%,#ecfeff_100%)] px-4 py-10 dark:bg-[radial-gradient(circle_at_20%_20%,#0e7490_0%,transparent_45%),radial-gradient(circle_at_80%_10%,#9f1239_0%,transparent_40%),linear-gradient(135deg,#020617_0%,#0f172a_55%,#082f49_100%)]">
      <Card className="w-full max-w-md animate-fade-up">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-xl">
            <img src="/autopilot-ai-icon.png" alt="AutoPilot AI" className="h-5 w-5" />
          </CardTitle>
          <CardDescription>
            Sign in to continue. AWS connection is mandatory for core analysis and operations.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={submit} className="space-y-3">
            <label className="text-xs font-medium text-slate-600 dark:text-slate-300">Email</label>
            <div className="relative">
              <Mail className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <Input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="pl-9" required />
            </div>
            <label className="text-xs font-medium text-slate-600 dark:text-slate-300">Password</label>
            <div className="relative">
              <LockKeyhole className="pointer-events-none absolute left-3 top-2.5 h-4 w-4 text-slate-400" />
              <Input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="pl-9" required minLength={8} />
            </div>
            {error && <p className="rounded-lg bg-rose-50 px-3 py-2 text-xs text-rose-700 dark:bg-rose-900/30 dark:text-rose-300">{error}</p>}
            <Button type="submit" className="w-full" disabled={loading}>
              {loading ? 'Please wait...' : mode === 'login' ? 'Sign In' : 'Create Account'}
            </Button>
          </form>
          <button
            className="mt-3 w-full text-sm text-cyan-700 hover:underline dark:text-cyan-300"
            onClick={() => setMode((prev) => (prev === 'login' ? 'register' : 'login'))}
          >
            {mode === 'login' ? 'Need an account? Register' : 'Already have an account? Login'}
          </button>
        </CardContent>
      </Card>
    </div>
  )
}

export default AuthPanel
