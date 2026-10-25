[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$RunDir,[switch]$RunModels)
$ErrorActionPreference='Continue'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$pythonArgs=@('-u',(Join-Path $PSScriptRoot 'run_qwen_ablation.py'),'--run-dir',$RunDir)
if ($RunModels) { $pythonArgs+='--run-models' }
& 'D:\conda\envs\agent_aml\python.exe' @pythonArgs 2>&1 | Tee-Object -Append -FilePath (Join-Path $RunDir 'console.log')
$resultCode=$LASTEXITCODE
Set-Content -LiteralPath (Join-Path $RunDir 'process_exit_code.txt') -Value $resultCode -Encoding Ascii
if ($RunModels -and $resultCode -eq 0) {
    & 'D:\conda\envs\agent_aml\python.exe' -u (Join-Path $PSScriptRoot 'verify_qwen_ablation.py') $RunDir 2>&1 | Tee-Object -Append -FilePath (Join-Path $RunDir 'verification.log')
    Set-Content -LiteralPath (Join-Path $RunDir 'verification_exit_code.txt') -Value $LASTEXITCODE -Encoding Ascii
}
Write-Host "QWEN_ABLATION_EXIT_CODE=$resultCode"
