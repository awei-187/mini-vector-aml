[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$RunDir)
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
& 'D:\conda\envs\agent_aml\python.exe' -u (Join-Path $PSScriptRoot 'run_optimization.py') --run-dir $RunDir 2>&1 | Tee-Object -Append -FilePath (Join-Path $RunDir 'console.log')
$pythonExitCode = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $RunDir 'process_exit_code.txt') -Value $pythonExitCode -Encoding Ascii
Write-Host "OPTIMIZATION_EXIT_CODE=$pythonExitCode"
