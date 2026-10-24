[CmdletBinding()]
param([Parameter(Mandatory=$true)][string]$RunDir,[switch]$RunModels)
$ErrorActionPreference = 'Continue'
Set-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null
$env:PYTHONUNBUFFERED='1'
$env:PYTHONIOENCODING='utf-8'
$pythonArgs=@('-u',(Join-Path $PSScriptRoot 'run_edited_dual.py'),'--run-dir',$RunDir)
if ($RunModels) { $pythonArgs+='--run-models' }
& 'D:\conda\envs\agent_aml\python.exe' @pythonArgs 2>&1 | Tee-Object -Append -FilePath (Join-Path $RunDir 'console.log')
$resultCode=$LASTEXITCODE
Set-Content -LiteralPath (Join-Path $RunDir 'process_exit_code.txt') -Value $resultCode -Encoding Ascii
if ($RunModels -and $resultCode -eq 0) {
    & 'D:\conda\envs\agent_aml\python.exe' -u (Join-Path $PSScriptRoot 'verify_edited_dual.py') $RunDir 2>&1 | Tee-Object -Append -FilePath (Join-Path $RunDir 'verification.log')
    $verifyCode=$LASTEXITCODE
    Set-Content -LiteralPath (Join-Path $RunDir 'verification_exit_code.txt') -Value $verifyCode -Encoding Ascii
}
Write-Host "EDITED_DUAL_EXIT_CODE=$resultCode"
