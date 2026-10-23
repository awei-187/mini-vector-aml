[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$RunDir, [switch]$Resume, [switch]$Baseline)
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonArgs = @('-u', (Join-Path $PSScriptRoot 'run_locomo.py'), '--run-dir', $RunDir)
if ($Resume) { $pythonArgs += '--resume' }
if ($Baseline) { $pythonArgs += '--baseline' }
Write-Host "MiniVector original LoCoMo conv-26: $RunDir"
& 'D:\conda\envs\agent_aml\python.exe' @pythonArgs 2>&1 | Tee-Object -Append -FilePath (Join-Path $RunDir 'console.log')
$pythonExitCode = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $RunDir 'process_exit_code.txt') -Value $pythonExitCode -Encoding Ascii
Write-Host "PYTHON_EXIT_CODE=$pythonExitCode"
