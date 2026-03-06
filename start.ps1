#!/usr/bin/env pwsh
<#
start.ps1 - Windows PowerShell launcher for AutoPilot AI

Usage:
  .\start.ps1 --backend
  .\start.ps1 --full
  .\start.ps1 --frontend
  .\start.ps1 --prod
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet('--backend', '--full', '--frontend', '--prod')]
    [string]$Mode = '--backend'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BackendProcess = $null

function Write-Info([string]$Message) {
    Write-Host "[AutoPilot] $Message" -ForegroundColor Cyan
}

function Write-Success([string]$Message) {
    Write-Host "[AutoPilot] $Message" -ForegroundColor Green
}

function Write-Warn([string]$Message) {
    Write-Host "[AutoPilot] $Message" -ForegroundColor Yellow
}

function Fail([string]$Message) {
    Write-Host "[AutoPilot ERROR] $Message" -ForegroundColor Red
    exit 1
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Activate-Venv {
    $candidates = @('venv', 'env', 'envn', '.venv')
    foreach ($v in $candidates) {
        $activateScripts = @(
            (Join-Path $Root "$v\Scripts\Activate.ps1"),
            (Join-Path $Root "$v\bin\Activate.ps1")
        )
        foreach ($scriptPath in $activateScripts) {
            if (Test-Path $scriptPath) {
                . $scriptPath
                Write-Success "Python venv activated: $v"
                return
            }
        }
    }

    Write-Warn "No virtual environment found in: $($candidates -join ', ') - using system Python"
}

function Stop-ProcessTree([int]$Pid) {
    $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $Pid" -ErrorAction SilentlyContinue
    foreach ($child in $children) {
        Stop-ProcessTree -Pid $child.ProcessId
    }

    $proc = Get-Process -Id $Pid -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $Pid -Force -ErrorAction SilentlyContinue
    }
}

function Start-Backend {
    Write-Info 'Starting FastAPI backend on http://0.0.0.0:8000'
    Write-Info '  API docs : http://localhost:8000/docs'
    Write-Info '  Health   : http://localhost:8000/api/health'
    Write-Host ''

    Push-Location $Root
    try {
        if (-not (Test-Command 'python')) {
            Fail 'Python not found in PATH. Activate a virtual environment or install Python.'
        }

        & python -m uvicorn autopilot_ai.api.main:app --host 0.0.0.0 --port 8000 --reload --reload-include '*.py' --log-level warning
    }
    finally {
        Pop-Location
    }
}

function Start-Frontend {
    if (-not (Test-Command 'node')) {
        Fail 'Node.js not found. Install Node.js 20+ and retry.'
    }

    Write-Info 'Starting Vite dev server on http://localhost:5173'
    Push-Location (Join-Path $Root 'frontend')
    try {
        if (-not (Test-Path 'node_modules')) {
            Write-Info 'Installing npm packages...'
            & npm install
        }

        & npm run dev
    }
    finally {
        Pop-Location
    }
}

function Build-Frontend {
    if (-not (Test-Command 'node')) {
        Fail 'Node.js not found. Cannot build frontend.'
    }

    Write-Info 'Building frontend for production...'
    Push-Location (Join-Path $Root 'frontend')
    try {
        if (-not (Test-Path 'node_modules')) {
            & npm install
        }
        & npm run build
        Write-Success 'Frontend built -> frontend/dist/'
    }
    finally {
        Pop-Location
    }
}

function Start-FullStack {
    Write-Info 'Starting FULL stack (backend + frontend dev server)'
    Write-Host ''

    Push-Location $Root
    try {
        $backendArgs = @(
            '-m', 'uvicorn',
            'autopilot_ai.api.main:app',
            '--host', '0.0.0.0',
            '--port', '8000',
            '--reload',
            '--log-level', 'warning'
        )

        $BackendProcess = Start-Process -FilePath 'python' -ArgumentList $backendArgs -PassThru -NoNewWindow
        Write-Info "Backend PID $($BackendProcess.Id) - waiting for readiness..."

        $ready = $false
        for ($i = 0; $i -lt 5; $i++) {
            Start-Sleep -Milliseconds 2000
            try {
                $null = Invoke-WebRequest -Uri 'http://localhost:8000/api/health' -Method Get -TimeoutSec 2
                $ready = $true
                break
            }
            catch {
                # Wait until backend responds.
            }
        }

        if ($ready) {
            Write-Success 'Backend ready!'
        }
        else {
            Write-Warn 'Backend did not report healthy within 10 seconds; starting frontend anyway.'
        }

        Start-Frontend
    }
    finally {
        Pop-Location
        if ($BackendProcess) {
            Stop-ProcessTree -Pid $BackendProcess.Id
        }
    }
}

if (-not (Test-Path (Join-Path $Root '.env'))) {
    Write-Warn '.env not found - copy .env.example and fill in your credentials.'
    Write-Warn 'Continuing with environment variables already set (if any).'
}

Activate-Venv

switch ($Mode) {
    '--backend' { Start-Backend }
    '--frontend' { Start-Frontend }
    '--full' { Start-FullStack }
    '--prod' {
        Build-Frontend
        Start-Backend
    }
}
