[CmdletBinding()]
param([Parameter(Mandatory = $true)][string]$RunDir, [string]$Baseline = 'E:\AML\mini-vector\eval_outputs\locomo_conv26_optimization_20261024_01\rrf', [switch]$RunModels)
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$env:PYTHONUNBUFFERED = '1'
$env:PYTHONIOENCODING = 'utf-8'
$pythonArgs = @('-u', (Join-Path $PSScriptRoot 'run_temporal_source.py'), '--baseline', $Baseline, '--run-dir', $RunDir)
if ($RunModels) { $pythonArgs += '--run-models' }
& 'D:\conda\envs\agent_aml\python.exe' @pythonArgs 2>&1 | Tee-Object -Append -FilePath (Join-Path $RunDir 'console.log')
$pythonExitCode = $LASTEXITCODE
Set-Content -LiteralPath (Join-Path $RunDir 'process_exit_code.txt') -Value $pythonExitCode -Encoding Ascii
Write-Host "TEMPORAL_SOURCE_EXIT_CODE=$pythonExitCode"
