# AlphaBattle

Post-training a small base LLM (`Qwen3-1.7B-Base`) into a competitive Pokémon Showdown (`gen9randombattle`) battling agent on a single consumer **RTX 3080 12GB** running **Native Windows**.

See [`docs/DESIGN.md`](docs/DESIGN.md) for the full technical design, hardware/VRAM feasibility math, verified 2026 prior-work & license audit, architecture rationale, staged SFT → Value Warm-Start → PPO training pipeline, and weekend milestone plan.

---

## Prerequisites (Native Windows PowerShell)
1. **Python 3.12** (or 3.11 — avoid 3.13/3.14 for Windows CUDA/`bitsandbytes` wheel stability)
2. **Node.js 22+ LTS** (Node 24 LTS recommended; `smogon/pokemon-showdown` strictly requires Node 22+)
3. **Git**
4. **Stage 0 Local LLM Server:** [Ollama](https://ollama.com/) (easiest) or prebuilt Windows CUDA [`llama.cpp` (`llama-server.exe`)](https://github.com/ggml-org/llama.cpp/releases) (fastest, supports 1-token GBNF grammar)

---

## Milestone 1: Local Showdown + Eval Harness + Stage 0 Baseline

### 1. Create the Python Environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
```

### 2. Set Up & Start the Local Pokémon Showdown Server
```powershell
.\scripts\setup_showdown.ps1
.\scripts\start_showdown.ps1 -Port 8000
```
Leave that PowerShell window running.

### 3. Smoke-Test the Eval Harness (Zero GPU)
```powershell
python scripts/eval.py --backend random --games 20 --concurrency 8 --output results/smoke_random.json
```

### 4. Run Your Stage 0 Prompted Baseline
**Option A — `llama-server` (1-token GBNF constrained decoding):**
```powershell
llama-server.exe -m Qwen3-4B-Q4_K_M.gguf --port 8080 -ngl 99 --ctx-size 4096 --parallel 4

python scripts/eval.py `
  --backend openai `
  --base-url http://127.0.0.1:8080/v1 `
  --model qwen3-4b `
  --mode direct `
  --constrain gbnf `
  --games 100 `
  --concurrency 4 `
  --output results/stage0_qwen3_4b_direct.json
```

**Option B — Ollama (JSON Schema constrained decoding):**
```powershell
ollama pull qwen3:4b
python scripts/eval.py `
  --backend openai `
  --base-url http://127.0.0.1:11434/v1 `
  --model qwen3:4b `
  --mode direct `
  --constrain json_schema `
  --games 100 `
  --concurrency 4 `
  --output results/stage0_ollama_qwen3_4b_direct.json
```

---

## Milestone 2: Zero-Gap Heuristic SFT + Direct PyTorch Masked-Logit Eval

### 1. Install PyTorch CUDA + Training Extras (Native Windows)
Install CUDA PyTorch **first** from [pytorch.org](https://pytorch.org/get-started/locally/), then install the `train` extras:
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -e ".[dev,train]"
```

### 2. Record 5,000 Zero-Sim2Sim-Gap Heuristic Mirror Games (~250K Turns)
```powershell
python scripts/record_trajectories.py `
  --games 5000 `
  --concurrency 16 `
  --teacher heuristics `
  --opponent heuristics `
  --record-both `
  --output data/m2_heuristics_5k.jsonl `
  --overwrite
```

### 3. Train `Qwen/Qwen3-1.7B-Base` bf16 LoRA (~1–1.5h, ~5.5–6.5 GiB Peak VRAM)
```powershell
python scripts/train_sft.py `
  --data data/m2_heuristics_5k.jsonl `
  --base-model Qwen/Qwen3-1.7B-Base `
  --output-dir checkpoints/m2_heuristics_sft `
  --epochs 1 `
  --batch-size 8 `
  --grad-accum 8 `
  --lr 2e-4 `
  --eval-every 200
```

### 4. Evaluate Trained Checkpoint with Batched PyTorch Masked-Logit Inference
```powershell
python scripts/eval.py `
  --backend torch `
  --checkpoint checkpoints/m2_heuristics_sft `
  --games 100 `
  --concurrency 16 `
  --max-batch-size 16 `
  --output results/m2_heuristics_sft_eval.json
```

---

## Milestones 3 & 4: High-Elo Human Replay SFT + Value Head Warm-Start

### 1. Build High-Elo Gen 9 Random Battle Dataset (Two-Pass Spectator-to-POV)
```powershell
# Saves BOTH sides (~50% win / ~50% loss) so Stage 1.5 Value Head sees balanced outcomes:
python scripts/build_replay_dataset.py `
  --hf-dataset HolidayOugi/pokemon-showdown-replays `
  --hf-data-files "[Gen 9] RANDOMBATTLE_part11.parquet" "[Gen 9] RANDOMBATTLE_part10.parquet" "[Gen 9] RANDOMBATTLE_part9.parquet" `
  --format-filter randombattle `
  --gen-filter 9 `
  --min-rating 1750 `
  --max-turns 240000 `
  --output data/gen9randbats_1750plus_both.jsonl `
  --overwrite
```

### 2. Stage 1 Human SFT (~2h on RTX 3080 12GB)
```powershell
python scripts/train_sft.py `
  --data data/gen9randbats_1750plus_both.jsonl `
  --base-model Qwen/Qwen3-1.7B-Base `
  --output-dir checkpoints/m3_human_sft `
  --winner-only `
  --epochs 1 `
  --batch-size 8 `
  --grad-accum 8 `
  --lr 2e-4 `
  --eval-every 450
```

### 3. Stage 1.5 Value Head Pretraining (~30 min, Policy Frozen, Balanced Win/Loss)
```powershell
python scripts/train_sft.py `
  --data data/gen9randbats_1750plus_both.jsonl `
  --resume-checkpoint checkpoints/m3_human_sft `
  --output-dir checkpoints/m4_sft_with_critic `
  --freeze-policy `
  --policy-loss-weight 0.0 `
  --value-loss-weight 1.0 `
  --require-known-outcome `
  --epochs 1 `
  --batch-size 16 `
  --grad-accum 4 `
  --lr 1e-3
```

### 4. Head-to-Head Checkpoint Evaluation
```powershell
python scripts/eval.py `
  --backend torch `
  --checkpoint checkpoints/m3_human_sft `
  --opponents heuristics `
  --opponent-checkpoint checkpoints/m2_heuristics_sft `
  --games 200 `
  --concurrency 16 `
  --output results/m3_vs_m2.json
```

---

## Milestone 5: Stage 2 Online Masked-Action PPO (Mixed Opponent Pool)

Start 2 local Showdown servers to utilize your i7-10700K cleanly, then run a 2–3h PPO session with the zero-extra-VRAM `"sft_ref"` KL anchor and rotating `"past_opp"` snapshots:
```powershell
# Terminal 1: start 2 local Showdown servers on ports 8000 and 8001
.\scripts\start_showdown.ps1 -Port 8000 -Instances 2

# Terminal 2: 100 PPO updates (~6,400+ battles across mixed pool)
python scripts/train_ppo.py `
  --init-checkpoint checkpoints/m4_sft_with_critic `
  --output-dir checkpoints/m5_ppo_v1 `
  --showdown-ports 8000 8001 `
  --updates 100 `
  --battles-per-update 64 `
  --concurrency 16 `
  --lr 2e-5 `
  --kl-coef 0.03 `
  --shaping-scale 1.0 `
  --anneal-shaping

# Evaluate RL checkpoint head-to-head against Stage 1 SFT and heuristics
python scripts/eval.py `
  --backend torch `
  --checkpoint checkpoints/m5_ppo_v1 `
  --opponents random max_power heuristics `
  --opponent-checkpoint checkpoints/m3_human_sft `
  --games 200 `
  --concurrency 16 `
  --output results/m5_ppo_v1_eval.json
```

---

## Milestone 6: Pre-Registered Public Showdown Ladder & Human Challenges

```powershell
# Safe mode first: accept challenges from your own Showdown account
python scripts/run_ladder.py `
  --mode accept --games 5 --allow-opponents YourShowdownUsername `
  --username AlphaBattleBot --password $env:PS_PASSWORD `
  --backend torch --checkpoint checkpoints/m5_ppo_v1

# Pre-registered 100-game rated ladder run (only after beating SimpleHeuristics locally!)
python scripts/run_ladder.py `
  --mode ladder --ladder-ack --games 100 `
  --username AlphaBattleBot --password $env:PS_PASSWORD `
  --backend torch --checkpoint checkpoints/m5_ppo_v1 `
  --output results/ladder_m5_ppo_v1.json
```

---

## Repository Layout
```text
alphabattle/
├── docs/DESIGN.md                   # Full technical design, VRAM math, prior work, RL plan
├── scripts/
│   ├── setup_showdown.ps1           # Native Windows Showdown installer & config patcher
│   ├── start_showdown.ps1           # Single or multi-instance local Showdown launcher
│   ├── setup_showdown.sh            # POSIX equivalent for Linux/CI
│   ├── eval.py                      # Fixed evaluation harness (OpenAI, PyTorch, checkpoint vs ckpt)
│   ├── record_trajectories.py       # Zero-sim2sim-gap local Showdown trajectory recorder
│   ├── build_replay_dataset.py      # HF / local Showdown spectator-to-POV dataset builder
│   ├── train_sft.py                 # 12GB LoRA SFT (Stage 1) & Value Head warm-start (Stage 1.5)
│   ├── train_ppo.py                 # Stage 2 Online Masked-Action PPO + mixed-pool self-play
│   └── run_ladder.py                # Milestone 6 gated public ladder & challenge runner
├── src/alphabattle/
│   ├── state.py                     # Compact (~475 tok) formatter + 13-ID ActionSpace + history
│   ├── backends.py                  # OpenAICompatBackend (JSON Schema / GBNF) + test backends
│   ├── torch_backend.py             # Async GPU micro-batched PyTorchPolicyBackend
│   ├── modeling.py                  # Backbone-sliced 13-way Policy + ValueHead + LoRA save/load
│   ├── ppo.py                       # GAE, potential shaping, dual-adapter KL anchor, PPO engine
│   ├── dataset.py                   # Canonical JSONL loader, battle-ID split, length bucketing
│   ├── recorder.py                  # Live poke-env trajectory recording mixins & players
│   ├── replays.py                   # Two-pass spectator-to-POV Showdown replay reconstructor
│   ├── player.py                    # Async poke-env 0.17 Player with strict legality & telemetry
│   └── metrics.py                   # Wilson CI, z-test, Elo delta, Brier score, ROC-AUC
└── tests/                           # Unit & end-to-end PyTorch/Qwen3/PEFT/poke-env/PPO tests
```
