<#
.SYNOPSIS
    Clone, configure, and build a local Pokémon Showdown server on Native Windows.
.DESCRIPTION
    Requires Git and Node.js 22+ (Node 22 or 24 LTS recommended by smogon/pokemon-showdown).
    Configures noguestsecurity, nothrottle, and noipchecks for high-throughput local eval & RL.
#>
param(
    [string]$TargetDir = "pokemon-showdown"
)

$ErrorActionPreference = "Stop"

# Verify Node.js >= 22
$nodeVersionRaw = (node -v).Trim()
$nodeMajor = [int]($nodeVersionRaw.TrimStart('v').Split('.')[0])
if ($nodeMajor -lt 22) {
    throw "Pokémon Showdown requires Node.js 22+ (24 LTS recommended). Found $nodeVersionRaw."
}
Write-Host "Using Node.js $nodeVersionRaw" -ForegroundColor Cyan

if (-not (Test-Path $TargetDir)) {
    Write-Host "Cloning smogon/pokemon-showdown into $TargetDir ..." -ForegroundColor Cyan
    git clone --depth 1 https://github.com/smogon/pokemon-showdown.git $TargetDir
} else {
    Write-Host "$TargetDir already exists; skipping clone." -ForegroundColor Yellow
}

Push-Location $TargetDir
try {
    Write-Host "Installing npm dependencies..." -ForegroundColor Cyan
    npm install

    $configPath = Join-Path "config" "config.js"
    $examplePath = Join-Path "config" "config-example.js"
    if (-not (Test-Path $configPath)) {
        Copy-Item $examplePath $configPath
    }

    $cfg = Get-Content $configPath -Raw
    $cfg = $cfg -replace "exports\.noguestsecurity\s*=\s*false;", "exports.noguestsecurity = true;"
    $cfg = $cfg -replace "exports\.nothrottle\s*=\s*false;", "exports.nothrottle = true;"
    $cfg = $cfg -replace "exports\.noipchecks\s*=\s*false;", "exports.noipchecks = true;"
    Set-Content -Path $configPath -Value $cfg -Encoding UTF8

    Write-Host "Building Pokémon Showdown TypeScript sources..." -ForegroundColor Cyan
    node build

    Write-Host ""
    Write-Host "Local Pokémon Showdown server is ready!" -ForegroundColor Green
    Write-Host "Start it with: .\scripts\start_showdown.ps1 -Port 8000" -ForegroundColor Green
} finally {
    Pop-Location
}
