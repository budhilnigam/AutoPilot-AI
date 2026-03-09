import React, { useState } from 'react'
import axios from 'axios'
import { CheckCircle2, Github, KeyRound, Link2Off, ShieldAlert, Copy, ChevronDown, ChevronUp } from 'lucide-react'
import { Button } from './ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Input } from './ui/input'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const AWS_IAM_POLICY = {
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AutoPilotAIRequiredPermissions",
      "Effect": "Allow",
      "Action": [
        "sts:GetCallerIdentity",
        "cloudwatch:ListMetrics",
        "cloudwatch:GetMetricData",
        "cloudwatch:GetMetricStatistics",
        "cloudwatch:DescribeAlarms",
        "cloudwatch:DescribeAlarmsForMetric",
        "logs:DescribeLogGroups",
        "logs:DescribeLogStreams",
        "logs:FilterLogEvents",
        "logs:StartQuery",
        "logs:GetQueryResults",
        "ce:GetCostAndUsage",
        "ce:GetCostForecast",
        "ce:GetDimensionValues",
        "ec2:DescribeInstances",
        "ec2:DescribeRegions",
        "ec2:DescribeVolumes",
        "ec2:DescribeSnapshots",
        "rds:DescribeDBInstances",
        "rds:DescribeDBClusters",
        "lambda:ListFunctions",
        "lambda:GetFunction",
        "ecs:ListClusters",
        "ecs:ListServices",
        "ecs:DescribeServices",
        "s3:ListAllMyBuckets",
        "s3:GetBucketLocation",
        "s3:GetObject",
        "s3:PutObject",
        "s3:ListBucket",
        "elasticloadbalancing:DescribeLoadBalancers",
        "elasticloadbalancing:DescribeTargetGroups",
        "cloudformation:DescribeStacks",
        "cloudformation:ListStacks",
        "sns:Publish",
        "bedrock:InvokeModel",
        "bedrock:Retrieve"
      ],
      "Resource": "*"
    }
  ]
}

function AccountConnections({ token, connectionStatus, onRefreshConnections }) {
  const [awsForm, setAwsForm] = useState({ accessKeyId: '', secretAccessKey: '', sessionToken: '', region: 'us-east-1' })
  const [ghForm, setGhForm] = useState({ repoOwner: '', repoName: '' })
  const [busy, setBusy] = useState('')
  const [message, setMessage] = useState('')
  const [showIamPolicy, setShowIamPolicy] = useState(false)
  const [copySuccess, setCopySuccess] = useState(false)

  const authHeaders = { Authorization: `Bearer ${token}` }

  const copyIamPolicy = () => {
    navigator.clipboard.writeText(JSON.stringify(AWS_IAM_POLICY, null, 2))
    setCopySuccess(true)
    setTimeout(() => setCopySuccess(false), 2000)
  }

  const connectAws = async (e) => {
    e.preventDefault()
    setBusy('aws')
    setMessage('')
    try {
      const res = await axios.post(`${API_BASE_URL}/api/auth/connect/aws`, {
        access_key_id: awsForm.accessKeyId,
        secret_access_key: awsForm.secretAccessKey,
        session_token: awsForm.sessionToken || null,
        region: awsForm.region,
      }, { headers: authHeaders })
      const missing = (res.data.missing_permissions || []).join(', ')
      setMessage(missing ? `AWS connected. Missing permissions: ${missing}` : 'AWS connected with all required permissions.')
      onRefreshConnections()
    } catch (err) {
      setMessage(err?.response?.data?.detail || 'AWS connection failed')
    } finally {
      setBusy('')
    }
  }

  const connectGithub = async (e) => {
    e.preventDefault()
    setBusy('github')
    setMessage('')
    try {
      const res = await axios.post(`${API_BASE_URL}/api/auth/connect/github/oauth/start`, {
        repo_owner: ghForm.repoOwner || null,
        repo_name: ghForm.repoName || null,
      }, { headers: authHeaders })

      const popup = window.open(res.data.auth_url, 'github-oauth', 'width=640,height=760')
      if (!popup) {
        setMessage('Unable to open GitHub OAuth popup. Please allow popups and try again.')
        return
      }

      const listener = (event) => {
        const payload = event?.data
        if (!payload || payload.type !== 'github-oauth') {
          return
        }
        window.removeEventListener('message', listener)
        if (payload.status === 'success') {
          setMessage('GitHub account connected successfully.')
          onRefreshConnections()
        } else {
          setMessage(payload.message || 'GitHub OAuth failed')
        }
      }

      window.addEventListener('message', listener)
    } catch (err) {
      setMessage(err?.response?.data?.detail || 'GitHub connection failed')
    } finally {
      setBusy('')
    }
  }

  const disconnectAws = async () => {
    setBusy('aws-disconnect')
    setMessage('')
    try {
      await axios.delete(`${API_BASE_URL}/api/auth/connect/aws`, { headers: authHeaders })
      setMessage('AWS account disconnected.')
      onRefreshConnections()
    } catch (err) {
      setMessage(err?.response?.data?.detail || 'Failed to disconnect AWS account')
    } finally {
      setBusy('')
    }
  }

  const disconnectGithub = async () => {
    setBusy('github-disconnect')
    setMessage('')
    try {
      await axios.delete(`${API_BASE_URL}/api/auth/connect/github`, { headers: authHeaders })
      setMessage('GitHub account disconnected.')
      onRefreshConnections()
    } catch (err) {
      setMessage(err?.response?.data?.detail || 'Failed to disconnect GitHub account')
    } finally {
      setBusy('')
    }
  }

  const updateGithubRepo = async () => {
    setBusy('github-repo')
    setMessage('')
    try {
      await axios.post(`${API_BASE_URL}/api/auth/connect/github/repository`, {
        repo_owner: ghForm.repoOwner || null,
        repo_name: ghForm.repoName || null,
      }, { headers: authHeaders })
      setMessage('GitHub repository preference updated.')
      onRefreshConnections()
    } catch (err) {
      setMessage(err?.response?.data?.detail || 'Failed to update GitHub repository')
    } finally {
      setBusy('')
    }
  }

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
      <Card className="animate-fade-up">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><KeyRound className="h-4 w-4 text-cyan-500" />Connect AWS (Mandatory)</CardTitle>
          <CardDescription>Required permissions: `sts:GetCallerIdentity`, `cloudwatch:ListMetrics`, `ce:GetCostAndUsage`.</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="mb-3 rounded-lg border border-blue-200 bg-blue-50 p-3 dark:border-blue-800 dark:bg-blue-950">
            <button
              type="button"
              onClick={() => setShowIamPolicy(!showIamPolicy)}
              className="flex w-full items-center justify-between text-sm font-semibold text-blue-900 dark:text-blue-100"
            >
              <span className="flex items-center gap-2">
                <ShieldAlert className="h-4 w-4" />
                IAM Policy Template (Copy & Paste)
              </span>
              {showIamPolicy ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>
            {showIamPolicy && (
              <div className="mt-3 space-y-2">
                <p className="text-xs text-blue-800 dark:text-blue-200">
                  When creating your IAM user, attach this policy to grant all required permissions:
                </p>
                <div className="relative">
                  <pre className="max-h-60 overflow-auto rounded bg-slate-900 p-3 text-xs text-slate-100">
                    {JSON.stringify(AWS_IAM_POLICY, null, 2)}
                  </pre>
                  <Button
                    type="button"
                    size="sm"
                    variant="secondary"
                    onClick={copyIamPolicy}
                    className="absolute right-2 top-2"
                  >
                    <Copy className="h-3 w-3" />
                    {copySuccess ? 'Copied!' : 'Copy'}
                  </Button>
                </div>
                <p className="text-xs text-blue-700 dark:text-blue-300">
                  💡 <strong>Setup:</strong> IAM → Users → Create user → Attach policies directly → Create policy → JSON tab → Paste above
                </p>
              </div>
            )}
          </div>
          <form onSubmit={connectAws} className="space-y-2">
            <Input placeholder="AWS Access Key ID" value={awsForm.accessKeyId} onChange={(e) => setAwsForm((p) => ({ ...p, accessKeyId: e.target.value }))} required />
            <Input placeholder="AWS Secret Access Key" type="password" value={awsForm.secretAccessKey} onChange={(e) => setAwsForm((p) => ({ ...p, secretAccessKey: e.target.value }))} required />
            <Input placeholder="AWS Session Token (optional)" value={awsForm.sessionToken} onChange={(e) => setAwsForm((p) => ({ ...p, sessionToken: e.target.value }))} />
            <Input placeholder="Region" value={awsForm.region} onChange={(e) => setAwsForm((p) => ({ ...p, region: e.target.value }))} required />
            <Button type="submit" disabled={busy === 'aws'} className="w-full">
              {busy === 'aws' ? (connectionStatus?.aws_connected ? 'Updating...' : 'Connecting...') : (connectionStatus?.aws_connected ? 'Update AWS Connection' : 'Connect AWS')}
            </Button>
            {connectionStatus?.aws_connected && (
              <p className="text-xs text-slate-500 dark:text-slate-400">
                💡 Enter new credentials above to update your connection
              </p>
            )}
          </form>
          {connectionStatus?.aws_connected && (
            <div className="mt-2 flex items-center justify-between gap-2">
              <p className="flex items-center gap-1 text-xs text-emerald-700 dark:text-emerald-300"><CheckCircle2 className="h-4 w-4" /> Connected account {connectionStatus.aws?.account_id || 'N/A'} ({connectionStatus.aws?.region})</p>
              <Button variant="ghost" size="sm" onClick={disconnectAws} disabled={busy === 'aws-disconnect'}>
                <Link2Off className="h-3 w-3" /> Disconnect
              </Button>
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="animate-fade-up delay-150">
        <CardHeader>
          <CardTitle className="flex items-center gap-2"><Github className="h-4 w-4 text-slate-700 dark:text-slate-200" />Connect GitHub (Optional)</CardTitle>
          <CardDescription>Needed for CI/CD insights and repository timeline. OAuth minimal scopes policy: `repo`, `workflow`, `read:user`.</CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={connectGithub} className="space-y-2">
            <Input placeholder="Repository Owner (optional)" value={ghForm.repoOwner} onChange={(e) => setGhForm((p) => ({ ...p, repoOwner: e.target.value }))} />
            <Input placeholder="Repository Name (optional)" value={ghForm.repoName} onChange={(e) => setGhForm((p) => ({ ...p, repoName: e.target.value }))} />
            <Button type="submit" disabled={busy === 'github'} className="w-full">
              {busy === 'github' ? 'Opening OAuth...' : (connectionStatus?.github_connected ? 'Update GitHub Connection' : 'Connect GitHub with OAuth')}
            </Button>
            {connectionStatus?.github_connected && (
              <p className="text-xs text-slate-500 dark:text-slate-400">
                💡 Click above to re-authenticate and update your GitHub connection
              </p>
            )}
            {connectionStatus?.github_connected && (
              <div className="grid grid-cols-2 gap-2">
                <Button type="button" variant="secondary" onClick={updateGithubRepo} disabled={busy === 'github-repo'}>
                  Save Repo Preference
                </Button>
                <Button type="button" variant="ghost" onClick={disconnectGithub} disabled={busy === 'github-disconnect'}>
                  <Link2Off className="h-3 w-3" /> Disconnect
                </Button>
              </div>
            )}
          </form>
          <div className="mt-3 rounded-lg border border-slate-200 bg-slate-50 p-2 text-xs text-slate-700 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200">
            <p className="font-semibold">Minimal Scopes Policy</p>
            <p>`repo`: read repository metadata and commits/workflow context.</p>
            <p>`workflow`: read workflow run status for CI/CD insights.</p>
            <p>`read:user`: identify the connected GitHub account.</p>
          </div>
          {!connectionStatus?.github_connected && (
            <p className="mt-2 flex items-center gap-1 text-xs text-amber-700 dark:text-amber-300"><ShieldAlert className="h-4 w-4" /> CI/CD and GitHub timeline features stay disabled until connected.</p>
          )}
          {connectionStatus?.github_connected && (
            <p className="mt-2 flex items-center gap-1 text-xs text-emerald-700 dark:text-emerald-300"><CheckCircle2 className="h-4 w-4" /> Connected as {connectionStatus.github?.username || 'GitHub user'}</p>
          )}
        </CardContent>
      </Card>

      {message && <p className="lg:col-span-2 rounded-xl bg-slate-100 px-3 py-2 text-sm text-slate-700 dark:bg-slate-800 dark:text-slate-200">{message}</p>}
    </div>
  )
}

export default AccountConnections
