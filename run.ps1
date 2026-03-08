#!/usr/bin/env pwsh
# AutoPilot AI - Run Script for Windows PowerShell

Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  AutoPilot AI - Multi-Agent SRE System" -ForegroundColor Cyan
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""

# Check if .env exists
if (-not (Test-Path ".env")) {
    Write-Host "WARNING: .env file not found. Creating from .env.example..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host "Please edit .env with your configuration before continuing" -ForegroundColor Green
    Read-Host "Press Enter to continue"
}

# Check if frontend/.env exists
if (-not (Test-Path "frontend\.env")) {
    Write-Host "WARNING: frontend/.env not found. Creating from frontend/.env.example..." -ForegroundColor Yellow
    Copy-Item "frontend\.env.example" "frontend\.env"
}

Write-Host "Starting AutoPilot AI..." -ForegroundColor Green
Write-Host ""

# Check Python
Write-Host "Checking Python..." -ForegroundColor Cyan
try {
    $pythonVersion = python --version 2>&1
    Write-Host "OK: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python not found. Please install Python 3.11+" -ForegroundColor Red
    exit 1
}

# Check Node.js
Write-Host "Checking Node.js..." -ForegroundColor Cyan
try {
    $nodeVersion = node --version 2>&1
    Write-Host "OK: Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Node.js not found. Please install Node.js 18+" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Installing dependencies..." -ForegroundColor Cyan

# Install Python dependencies
Write-Host "Installing Python packages..." -ForegroundColor Yellow
pip install -q -r requirements.txt
if ($LASTEXITCODE -eq 0) {
    Write-Host "Python dependencies installed" -ForegroundColor Green
} else {
    Write-Host "Warning: Some Python dependencies may have failed" -ForegroundColor Yellow
}

# Install Node packages
Write-Host "Installing Node.js packages..." -ForegroundColor Yellow
Push-Location frontend
npm install --silent
if ($LASTEXITCODE -eq 0) {
    Write-Host "Node.js dependencies installed" -ForegroundColor Green
} else {
    Write-Host "Warning: Some Node.js dependencies may have failed" -ForegroundColor Yellow
}
Pop-Location

Write-Host ""
Write-Host "Starting services..." -ForegroundColor Cyan
Write-Host ""

$script:backend = $null
$script:frontend = $null

function Stop-Services {
    if ($script:backend -and -not $script:backend.HasExited) {
        Stop-Process -Id $script:backend.Id -Force -ErrorAction SilentlyContinue
    }
    if ($script:frontend -and -not $script:frontend.HasExited) {
        Stop-Process -Id $script:frontend.Id -Force -ErrorAction SilentlyContinue
    }
}

# Start backend in background (separate process for reliable Ctrl+C handling)
Write-Host "Starting backend on http://localhost:8000" -ForegroundColor Yellow
$script:backend = Start-Process python -ArgumentList "-m", "uvicorn", "app:app", "--reload", "--host", "0.0.0.0", "--port", "8000" -PassThru

Start-Sleep -Seconds 3

# Start frontend
Write-Host "Starting frontend on http://localhost:5173" -ForegroundColor Yellow
Push-Location frontend
$script:frontend = Start-Process npm -ArgumentList "run", "dev" -PassThru
Pop-Location

Write-Host ""
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host "  AutoPilot AI is running!" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Frontend: http://localhost:5173" -ForegroundColor White
Write-Host "Backend API: http://localhost:8000" -ForegroundColor White
Write-Host "API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
Write-Host "Press Ctrl+C to stop all services" -ForegroundColor Yellow
Write-Host ""

$script:ctrlCPressed = $false
trap [System.Management.Automation.PipelineStoppedException] {
    $script:ctrlCPressed = $true
    continue
}

# Wait until one process exits or Ctrl+C is pressed.
try {
    while ($true) {
        if ($script:backend.HasExited -or $script:frontend.HasExited) {
            break
        }
        Start-Sleep -Seconds 1
    }
} finally {
    Write-Host ""
    if ($script:ctrlCPressed) {
        Write-Host "Ctrl+C detected. Stopping services..." -ForegroundColor Yellow
    } else {
        Write-Host "A service exited. Stopping remaining services..." -ForegroundColor Yellow
    }
    Stop-Services
    Write-Host "Services stopped" -ForegroundColor Green
}
