param(
    [switch]$NoBrowser,
    [int]$Port = 7861
)

$ErrorActionPreference = 'Stop'
$portableRoot = $PSScriptRoot
$environmentRoot = Join-Path $portableRoot '.venv-host-gold'
$python = Join-Path $environmentRoot 'Scripts\python.exe'
$requirements = Join-Path $portableRoot 'requirements_host_gold.txt'
$application = Join-Path $portableRoot 'annotate_plantseg_host_gold.py'
$config = Join-Path $portableRoot 'host_train_annotation_config.json'

function Find-BasePython {
    # Try each launcher and keep the first one that really runs Python 3.10+.
    # 'py' can exist without any registered Python ("No installed Python found!"), and
    # 'python' can be the Microsoft Store stub, so every candidate is tested.
    $candidates = @(@('py', '-3'), @('python'), @('python3'))
    foreach ($candidate in $candidates) {
        $command = Get-Command $candidate[0] -ErrorAction SilentlyContinue
        if (-not $command) { continue }
        $prefix = @()
        if ($candidate.Count -gt 1) { $prefix = $candidate[1..($candidate.Count - 1)] }
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $ok = & $command.Source @prefix -c "import sys; print(1 if sys.version_info >= (3, 10) else 0)" 2>$null
        $ErrorActionPreference = $previous
        if ($LASTEXITCODE -eq 0 -and "$ok".Trim() -eq '1') {
            return @{ Path = $command.Source; Prefix = $prefix }
        }
    }
    return $null
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    Write-Host 'Creating the portable Python environment...'
    $base = Find-BasePython
    if (-not $base) {
        throw 'Python 3.10 or newer is required. Install it from python.org (tick "Add python.exe to PATH"), then run this script again.'
    }
    $basePrefix = $base.Prefix
    & $base.Path @basePrefix -m venv $environmentRoot
    if ($LASTEXITCODE -ne 0) { throw 'Could not create the Python environment.' }
    & $python -m pip install --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Could not install Pillow and NumPy.' }
}

$arguments = @($application, '--config', $config, '--port', $Port)
if ($NoBrowser) { $arguments += '--no-browser' }

Push-Location $portableRoot
try {
    & $python @arguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
