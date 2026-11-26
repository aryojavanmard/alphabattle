<#
.SYNOPSIS
    Start local Pokémon Showdown server(s) on Native Windows with --no-security.
.EXAMPLE
    .\scripts\start_showdown.ps1 -Port 8000
.EXAMPLE
    .\scripts\start_showdown.ps1 -Port 8000 -Instances 2
#>
param(
    [string]$ServerDir = "pokemon-showdown",
    [int]$Port = 8000,
    [int]$Instances = 1
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $ServerDir)) {
    throw "Directory '$ServerDir' not found. Run .\scripts\setup_showdown.ps1 first."
}

if ($Instances -eq 1) {
    Write-Host "Starting local Pokémon Showdown server on ws://127.0.0.1:$Port/showdown/websocket ..." -ForegroundColor Cyan
    Push-Location $ServerDir
    try {
        node pokemon-showdown start --no-security --skip-build $Port
    } finally {
        Pop-Location
    }
} else {
    Write-Host "Launching $Instances Pokémon Showdown server instances on ports $Port..$($Port + $Instances - 1)..." -ForegroundColor Cyan
    for ($i = 0; $i -lt $Instances; $i++) {
        $p = $Port + $i
        Start-Process -FilePath "node" `
            -ArgumentList "pokemon-showdown", "start", "--no-security", "--skip-build", "$p" `
            -WorkingDirectory (Resolve-Path $ServerDir)
        Write-Host "  Started instance on port $p" -ForegroundColor Green
    }
}
