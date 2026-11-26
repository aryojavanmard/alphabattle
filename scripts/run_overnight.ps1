<#
.SYNOPSIS
    Run the full AlphaBattle pipeline sequentially (Data -> Stage 1 SFT -> Eval ->
    Stage 1.5 Critic -> Stage 2 PPO -> Final Eval) and log everything to a file.
.DESCRIPTION
    Designed for overnight runs on Native Windows PowerShell 5.1+ with an RTX 3080 12GB.
    Uses local Pokémon Showdown on Port 8000 (auto-detects an already-running server, or
    pass -StartShowdown to launch and stop one automatically).

.EXAMPLE
    # 1. Run a ~10-15 minute sanity check while awake first:
    .\scripts\run_overnight.ps1 -QuickTest -StartShowdown

.EXAMPLE
    # 2. Full first real Qwen3-1.7B-Base overnight run (~5-7 hours on RTX 3080 12GB):
    .\scripts\run_overnight.ps1 -StartShowdown
#>
param(
    [switch]$QuickTest,
    [switch]$StartShowdown,
    [int]$ShowdownPort = 8000,
    [string]$BaseModel = "Qwen/Qwen3-1.7B-Base",
    [int]$MinRating = 1750,
    [int]$MaxReplayTurns = 240000,
    [int]$PPOUpdates = 60,
    [int]$EvalGames = 100
)

$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"

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

function Test-ShowdownPort {
    param([int]$Port)
    try {
        $resp = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/" -TimeoutSec 2 -UseBasicParsing
        return ($resp.StatusCode -ge 200)
    } catch {
        return $false
    }
}

function Run-Stage {
    param(
        [string]$Title,
        [string[]]$PyArgs
    )
    $banner = "`n============================================================`n[$(Get-Date -Format 'HH:mm:ss')] STARTING: $Title`n============================================================"
    $banner | Tee-Object -FilePath $logFile -Append

    # Windows PowerShell 5.1 turns native stderr (HF warnings, tqdm, poke-env logs) into
    # terminating ErrorRecords when $ErrorActionPreference is "Stop". Temporarily switch to
    # "Continue", stringify every line cleanly, and enforce real failures via $LASTEXITCODE.
    $prevEAP = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    $exitCode = 0
    try {
        & $pythonExe -u @PyArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $logFile -Append
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prevEAP
    }

    if ($exitCode -ne 0) {
        throw "Stage '$Title' failed with exit code $exitCode. See $logFile"
    }
    "[$(Get-Date -Format 'HH:mm:ss')] COMPLETED: $Title" | Tee-Object -FilePath $logFile -Append
}

$showdownProc = $null
try {
    if (Test-ShowdownPort -Port $ShowdownPort) {
        "Detected running Pokémon Showdown server on port $ShowdownPort (reusing it)." | Tee-Object -FilePath $logFile -Append
    } elseif ($StartShowdown) {
        if (-not (Test-Path "pokemon-showdown")) {
            Write-Host "Setting up local Pokémon Showdown server..." -ForegroundColor Cyan
            & ".\scripts\setup_showdown.ps1"
        }
        Write-Host "Starting background Pokémon Showdown server on port $ShowdownPort..." -ForegroundColor Cyan
        $showdownProc = Start-Process -FilePath "node" `
            -ArgumentList "pokemon-showdown", "start", "--no-security", "--skip-build", "$ShowdownPort" `
            -WorkingDirectory (Resolve-Path "pokemon-showdown") `
            -PassThru
        $ready = $false
        for ($i = 0; $i -lt 30; $i++) {
            if (Test-ShowdownPort -Port $ShowdownPort) { $ready = $true; break }
            Start-Sleep -Milliseconds 500
        }
        if (-not $ready) {
            throw "Pokémon Showdown did not become ready on port $ShowdownPort."
        }
        "Pokémon Showdown started on port $ShowdownPort (PID $($showdownProc.Id))." | Tee-Object -FilePath $logFile -Append
    } else {
        throw "No Pokémon Showdown server detected on port $ShowdownPort. Either start it with .\scripts\start_showdown.ps1 -Port $ShowdownPort or pass -StartShowdown."
    }

    if ($QuickTest) {
        # ~10-15 minute full pipeline sanity check with Qwen3-0.6B-Base
        $model = "Qwen/Qwen3-0.6B-Base"
        $dataPath = "data\quicktest_200.jsonl"
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
        # Full overnight run (~5-7 hours on RTX 3080 12GB)
        # Stage 1 saves BOTH sides (~50% win / ~50% loss) so Stage 1.5 Value Head sees balanced outcomes,
        # while Stage 2 filters with --winner-only so Policy SFT imitates only winning high-Elo players.
        $dataPath = "data\gen9randbats_${MinRating}plus_both.jsonl"
        $ckptS1 = "checkpoints\s1_human_sft"
        $ckptS15 = "checkpoints\s15_sft_critic"
        $ckptS2 = "checkpoints\s2_ppo_v1"

        if (-not (Test-Path $dataPath)) {
            Run-Stage "1/6 Stream & reconstruct $MaxReplayTurns high-Elo ($MinRating+) 2026 human turns" @(
                "scripts\build_replay_dataset.py",
                "--hf-dataset", "HolidayOugi/pokemon-showdown-replays",
                "--hf-data-files",
                "[Gen 9] RANDOMBATTLE_part11.parquet",
                "[Gen 9] RANDOMBATTLE_part10.parquet",
                "[Gen 9] RANDOMBATTLE_part9.parquet",
                "[Gen 9] RANDOMBATTLE_part8.parquet",
                "--format-filter", "randombattle", "--gen-filter", "9",
                "--min-rating", "$MinRating",
                "--max-turns", "$MaxReplayTurns",
                "--output", $dataPath, "--overwrite"
            )
        } else {
            "Skipping dataset build; $dataPath already exists." | Tee-Object -FilePath $logFile -Append
        }

        Run-Stage "2/6 Stage 1 Human Policy SFT ($BaseModel, winners only)" @(
            "scripts\train_sft.py",
            "--data", $dataPath, "--base-model", $BaseModel, "--output-dir", $ckptS1,
            "--winner-only", "--epochs", "1", "--batch-size", "8", "--grad-accum", "8", "--lr", "2e-4",
            "--eval-every", "450"
        )

        Run-Stage "3/6 Eval Stage 1 Human SFT ($EvalGames games/opponent)" @(
            "scripts\eval.py",
            "--backend", "torch", "--checkpoint", $ckptS1,
            "--opponents", "random", "max_power", "heuristics",
            "--games", "$EvalGames", "--concurrency", "16", "--showdown-port", "$ShowdownPort",
            "--output", "results\$runTag`_s1_eval.json"
        )

        Run-Stage "4/6 Stage 1.5 Value Head warm-start (balanced win/loss, policy frozen)" @(
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
            "--snapshot-every", "15", "--save-every", "10"
        )

        Run-Stage "6/6 Final Eval: Stage 2 PPO vs Baselines & Stage 1 SFT ($EvalGames games/opponent)" @(
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
