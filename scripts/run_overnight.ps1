<#
.SYNOPSIS
    Run the full AlphaBattle pipeline sequentially (Data -> Stage 1 SFT -> Eval ->
    Stage 1.5 Critic -> Stage 2 PPO -> Final Eval) and log everything to a file.
.DESCRIPTION
    Designed for overnight runs on Native Windows with an RTX 3080 12GB.
    Requires Pokémon Showdown already running on Port 8000 (or pass -StartShowdown
    to launch and stop it automatically).

.EXAMPLE
    # 1. Run a 10-minute sanity check while awake first:
    .\scripts\run_overnight.ps1 -QuickTest -StartShowdown

.EXAMPLE
    # 2. Full overnight run (~5-6 hours on RTX 3080 12GB):
    .\scripts\run_overnight.ps1 -StartShowdown
#>
param(
    [switch]$QuickTest,
    [switch]$StartShowdown,
    [int]$ShowdownPort = 8000,
    [string]$BaseModel = "Qwen/Qwen3-1.7B-Base",
    [int]$MaxReplayTurns = 250000,
    [int]$PPOUpdates = 100,
    [int]$EvalGames = 100
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

$pythonExe = Join-Path $repoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $pythonExe)) {
    throw "Virtual environment not found at $pythonExe. Activate/create .venv first."
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$runTag = if ($QuickTest) { "quicktest_$timestamp" } else { "overnight_$timestamp" }
New-Item -ItemType Directory -Force -Path "logs", "data", "checkpoints", "results" | Out-Null
$logFile = Join-Path "logs" "$runTag.log"

function Run-Stage {
    param(
        [string]$Title,
        [string[]]$PyArgs
    )
    $banner = "`n============================================================`n[$(Get-Date -Format 'HH:mm:ss')] STARTING: $Title`n============================================================"
    $banner | Tee-Object -FilePath $logFile -Append
    & $pythonExe @PyArgs 2>&1 | Tee-Object -FilePath $logFile -Append
    if ($LASTEXITCODE -ne 0) {
        throw "Stage '$Title' failed with exit code $LASTEXITCODE. See $logFile"
    }
    "[$(Get-Date -Format 'HH:mm:ss')] COMPLETED: $Title" | Tee-Object -FilePath $logFile -Append
}

$showdownProc = $null
try {
    if ($StartShowdown) {
        if (-not (Test-Path "pokemon-showdown")) {
            Write-Host "Setting up local Pokémon Showdown server..." -ForegroundColor Cyan
            & ".\scripts\setup_showdown.ps1"
        }
        Write-Host "Starting background Pokémon Showdown server on port $ShowdownPort..." -ForegroundColor Cyan
        $showdownProc = Start-Process -FilePath "node" `
            -ArgumentList "pokemon-showdown", "start", "--no-security", "--skip-build", "$ShowdownPort" `
            -WorkingDirectory (Resolve-Path "pokemon-showdown") `
            -PassThru
        # Wait up to 15s for Showdown to listen
        $ready = $false
        for ($i = 0; $i -lt 30; $i++) {
            try {
                $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$ShowdownPort/" -TimeoutSec 2 -UseBasicParsing
                if ($resp.StatusCode -ge 200) { $ready = $true; break }
            } catch {
                Start-Sleep -Milliseconds 500
            }
        }
        if (-not $ready) {
            throw "Pokémon Showdown did not become ready on port $ShowdownPort."
        }
        Write-Host "Pokémon Showdown is ready on port $ShowdownPort." -ForegroundColor Green
    }

    if ($QuickTest) {
        # ~10-15 minute full pipeline sanity check with Qwen3-0.6B-Base
        $model = "Qwen/Qwen3-0.6B-Base"
        $dataPath = "data\quicktest_500.jsonl"
        $ckptS1 = "checkpoints\qt_s1_sft"
        $ckptS15 = "checkpoints\qt_s15_critic"
        $ckptS2 = "checkpoints\qt_s2_ppo"

        Run-Stage "1/5 Record 200 local heuristic mirror battles" @(
            "scripts\record_trajectories.py",
            "--games", "200", "--concurrency", "16",
            "--teacher", "heuristics", "--opponent", "heuristics", "--record-both",
            "--showdown-port", "$ShowdownPort",
            "--output", $dataPath, "--overwrite"
        )
        Run-Stage "2/5 Stage 1 SFT ($model)" @(
            "scripts\train_sft.py",
            "--data", $dataPath, "--base-model", $model, "--output-dir", $ckptS1,
            "--epochs", "1", "--batch-size", "8", "--grad-accum", "8", "--lr", "2e-4"
        )
        Run-Stage "3/5 Stage 1.5 Value Head warm-start" @(
            "scripts\train_sft.py",
            "--data", $dataPath, "--resume-checkpoint", $ckptS1, "--output-dir", $ckptS15,
            "--freeze-policy", "--policy-loss-weight", "0.0", "--value-loss-weight", "1.0",
            "--require-known-outcome", "--epochs", "1", "--batch-size", "16", "--grad-accum", "4", "--lr", "1e-3"
        )
        Run-Stage "4/5 Stage 2 Online PPO (5 quick updates)" @(
            "scripts\train_ppo.py",
            "--init-checkpoint", $ckptS15, "--output-dir", $ckptS2,
            "--showdown-ports", "$ShowdownPort",
            "--updates", "5", "--battles-per-update", "16", "--concurrency", "8",
            "--lr", "2e-5", "--kl-coef", "0.03", "--anneal-shaping", "--snapshot-every", "2"
        )
        Run-Stage "5/5 Eval PPO vs Baselines & Stage 1 SFT" @(
            "scripts\eval.py",
            "--backend", "torch", "--checkpoint", $ckptS2,
            "--opponents", "random", "max_power", "heuristics",
            "--opponent-checkpoint", $ckptS1,
            "--games", "20", "--concurrency", "8", "--showdown-port", "$ShowdownPort",
            "--output", "results\$runTag`_eval.json"
        )
    } else {
        # Full overnight run (~5-6 hours on RTX 3080 12GB)
        $dataPath = "data\gen9randbats_1550plus.jsonl"
        $ckptS1 = "checkpoints\s1_human_sft"
        $ckptS15 = "checkpoints\s15_sft_critic"
        $ckptS2 = "checkpoints\s2_ppo_v1"

        if (-not (Test-Path $dataPath)) {
            Run-Stage "1/6 Stream & reconstruct $MaxReplayTurns high-Elo human turns" @(
                "scripts\build_replay_dataset.py",
                "--hf-dataset", "milkkarten/pokemon-showdown-replays-merged",
                "--format-filter", "randombattle", "--gen-filter", "9",
                "--min-rating", "1550", "--winner-only",
                "--max-turns", "$MaxReplayTurns",
                "--output", $dataPath, "--overwrite"
            )
        } else {
            "Skipping dataset build; $dataPath already exists." | Tee-Object -FilePath $logFile -Append
        }

        Run-Stage "2/6 Stage 1 Human SFT ($BaseModel)" @(
            "scripts\train_sft.py",
            "--data", $dataPath, "--base-model", $BaseModel, "--output-dir", $ckptS1,
            "--epochs", "1", "--batch-size", "8", "--grad-accum", "8", "--lr", "2e-4",
            "--eval-every", "250"
        )

        Run-Stage "3/6 Eval Stage 1 SFT" @(
            "scripts\eval.py",
            "--backend", "torch", "--checkpoint", $ckptS1,
            "--opponents", "random", "max_power", "heuristics",
            "--games", "$EvalGames", "--concurrency", "16", "--showdown-port", "$ShowdownPort",
            "--output", "results\$runTag`_s1_eval.json"
        )

        Run-Stage "4/6 Stage 1.5 Value Head warm-start" @(
            "scripts\train_sft.py",
            "--data", $dataPath, "--resume-checkpoint", $ckptS1, "--output-dir", $ckptS15,
            "--freeze-policy", "--policy-loss-weight", "0.0", "--value-loss-weight", "1.0",
            "--require-known-outcome", "--epochs", "1", "--batch-size", "16", "--grad-accum", "4", "--lr", "1e-3"
        )

        Run-Stage "5/6 Stage 2 Online Masked-Action PPO ($PPOUpdates updates)" @(
            "scripts\train_ppo.py",
            "--init-checkpoint", $ckptS15, "--output-dir", $ckptS2,
            "--showdown-ports", "$ShowdownPort",
            "--updates", "$PPOUpdates", "--battles-per-update", "64", "--concurrency", "16",
            "--lr", "2e-5", "--kl-coef", "0.03", "--shaping-scale", "1.0", "--anneal-shaping",
            "--snapshot-every", "20", "--save-every", "10"
        )

        Run-Stage "6/6 Final Eval: Stage 2 PPO vs Baselines & Stage 1 SFT" @(
            "scripts\eval.py",
            "--backend", "torch", "--checkpoint", $ckptS2,
            "--opponents", "random", "max_power", "heuristics",
            "--opponent-checkpoint", $ckptS1,
            "--games", "$EvalGames", "--concurrency", "16", "--showdown-port", "$ShowdownPort",
            "--output", "results\$runTag`_s2_eval.json"
        )
    }

    Write-Host "`nALL STAGES FINISHED! Full log saved to: $logFile" -ForegroundColor Green
} finally {
    if ($null -ne $showdownProc -and -not $showdownProc.HasExited) {
        Write-Host "Stopping background Pokémon Showdown server (PID $($showdownProc.Id))..." -ForegroundColor Yellow
        Stop-Process -Id $showdownProc.Id -Force -ErrorAction SilentlyContinue
    }
}
