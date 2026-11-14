$ErrorActionPreference = 'Stop'
$portableRoot = $PSScriptRoot
$python = Join-Path $portableRoot '.venv-host-gold\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw 'Run start_host_train.ps1 once to create the portable Python environment.'
}
& $python (Join-Path $portableRoot 'annotate_plantseg_host_gold.py') `
    --config (Join-Path $portableRoot 'host_train_annotation_config.json') `
    --status
exit $LASTEXITCODE
