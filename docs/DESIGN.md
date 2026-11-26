# AlphaBattle — Technical Design & Post-Training Blueprint

**Target Hardware:** RTX 3080 12GB (Ampere, `sm_86`, native `bf16`, no `FP8`), Intel i7-10700K (8C/16T), 32GB DDR4 RAM  
**Target OS:** Native Windows 10/11 (no WSL2)  
**Target Training Cadence:** Strictly local weekend bursts (1–3 hours per run)  
**Target Format:** `gen9randombattle` first, designed to extend cleanly to `gen9ou` (fixed teams) later  
**Primary Objective:** Post-train a small base LLM end-to-end (SFT → Value Warm-Start → Online RL) and measure strength honestly by win rate, Wilson 95% CIs, and Showdown Elo against heuristic, search, and human opponents.

---

## 1. Hardware & Native Windows Feasibility Audit

### 1.1 Effective VRAM Budget (The Hidden Windows Tax)
On Native Windows (WDDM display driver), Windows DWM, desktop apps, and PyTorch allocator fragmentation reserve **1.2–2.0 GB** of VRAM before your script allocates a single tensor.
- **Usable training VRAM ceiling:** **~10.0–10.5 GB** (not 12.0 GB).
- **Critical pitfall:** If PyTorch exceeds physical VRAM on Windows, NVIDIA's driver silently spills tensors into **Shared GPU Memory** (system DDR4 RAM over PCIe) instead of throwing `CUDA OOM`, causing a **10–30× slowdown**. Always verify `torch.cuda.max_memory_allocated() < 10.2 GiB` and watch Dedicated vs. Shared GPU memory in Task Manager / `nvidia-smi`.

### 1.2 Model Size × Training Method Feasibility Matrix (`seq_len <= 1024`)

Standard mixed-precision AdamW full fine-tuning requires ~16 bytes/param (`2B bf16 weights + 2B grads + 4B fp32 master + 8B Adam moments`). LoRA/QLoRA eliminates optimizer states on frozen base weights:

| Model Size (Example, Apache 2.0) | Full FT (8-bit AdamW + GC) | bf16 LoRA (`r=32`, 8-bit AdamW + GC) | QLoRA NF4 (`r=32`, 8-bit AdamW + GC) | Stage 2 Online RL Verdict |
|---|---|---|---|---|
| **0.6B** (`Qwen/Qwen3-0.6B-Base`) | ~5.5–6.5 GB (Fits) | ~2.8–3.5 GB (Easy) | ~2.0–2.5 GB (Easy) | **Best for fast 15-min pipeline debugging** |
| **1.7B** (`Qwen/Qwen3-1.7B-Base`) | ~14–16 GB (**OOM**) | **~6.5–8.0 GB (Sweet Spot)** | ~4.2–5.5 GB (Very comfortable) | **Recommended primary model** (fits bf16 LoRA + Value Head + 16-battle rollouts) |
| **3B–4B** (`Qwen/Qwen3-4B-Base`) | >26 GB (**OOM**) | ~9.5–11.5 GB (**Spills on Windows**) | ~6.0–7.5 GB (Fits SFT) | Good for Stage 0 prompted eval; ~2.5× slower RL than 1.7B |
| **7B–8B** (`Qwen/Qwen3-8B-Base`) | >50 GB (**OOM**) | >17 GB (**OOM**) | ~8.5–10.0 GB (Tight SFT only) | **Avoid for RL on 12GB Windows:** too slow for 1–3h runs, severe OOM/spill risk |

**Recommendation:** Use **`Qwen/Qwen3-1.7B-Base`** (Apache 2.0, 28 layers, GQA 16Q/8KV, matches Qwen2.5-3B on benchmarks) in **bf16 LoRA** (avoiding NF4 quantization error) for Stages 1–2, and **`Qwen/Qwen3-0.6B-Base`** as a fast sanity-check model. Use **`Qwen/Qwen3-4B`** (Q4_K_M via `llama-server`/Ollama) alongside `Qwen3-1.7B` for the Stage 0 prompted baseline.

### 1.3 Context Length & Throughput Reality
- **State prompt budget:** Our structured battle state + 3-turn history + legal actions is **~350–550 tokens**. Cap `max_seq_len = 1024`. Full raw battle transcripts (3K–8K tokens) are unnecessary and waste quadratic attention compute.
- **Expected RTX 3080 12GB throughput (~912 GB/s bandwidth, ~119 TFLOPS bf16):**
  - **SFT (`Qwen3-1.7B-Base`, bf16 LoRA, `seq_len<=1024`):** ~4,500–7,000 tok/s → **~25K–45K battle turns/hour**. A 2-hour weekend run easily trains 1–2 epochs on 150K–250K high-Elo turns.
  - **Inference / RL Rollout (`Qwen3-1.7B-Base`, bf16, Direct Single-Token Action):** One batched prefill of `batch=16 × 500 tokens` takes ~40–60 ms → **~250–400 decisions/second** → **~8–13 full 30-turn battles/second** (~30K–45K battles/hour!).
  - **Inference / RL Rollout (100-token CoT mode):** Autoregressive decoding drops throughput to **~6–15 decisions/second** → **~0.2–0.5 battles/second** (~700–1,500 battles/hour) — a **30–50× slowdown**.

### 1.4 Native Windows Tooling Flags (Verified Nov 2026)
- **Do NOT waste time installing natively on Windows:**
  - `vLLM`, `SGLang`, `DeepSpeed`: Linux/WSL2 only.
  - `flash-attn` pip builds: brittle C++/NVCC builds on Windows; **use PyTorch built-in `attn_implementation="sdpa"`**, which dispatches FlashAttention-2 on Ampere CUDA 12.x automatically with zero compilation.
  - `Python 3.13 / 3.14`: lags CUDA/bitsandbytes/Triton Windows wheel availability — **pin Python 3.12 (or 3.11)**.
- **Works cleanly natively on Windows:**
  - **Stage 0 fast local serving:** Prebuilt Windows CUDA binaries of [`llama.cpp` (`llama-server.exe`)](https://github.com/ggml-org/llama.cpp/releases) (supports `--parallel N` continuous batching + 1-token GBNF grammar) or **Ollama** (JSON Schema constrained outputs).
  - **Stage 1–2 training & in-process RL:** `PyTorch` (CUDA 12.4/12.6) + `transformers` + `peft` + `trl` + `bitsandbytes` (official native Windows CUDA wheels since `>=0.45`) with `attn_implementation="sdpa"`. Optional: `Unsloth` (native Windows support via `triton-windows`; requires `dataset_num_proc=1` due to Windows `spawn` multiprocessing).
  - **Local Showdown server:** Node.js **22+ LTS** (24 LTS recommended — Showdown now strictly checks `Set.prototype.intersection`, rejecting Node <22!).

---

## 2. Prior Work Survey (Verified State & Licenses, Nov 2026)

| Project / Paper | Repo / Paper | Current State (Nov 2026) | Verified License | What to Reuse / Key Takeaway |
|---|---|---|---|---|
| **Pokémon Showdown** | [smogon/pokemon-showdown](https://github.com/smogon/pokemon-showdown) | Active daily (Node 22+ required) | **MIT** (server); client is **AGPL-3.0** | **Reuse directly** locally with `--no-security`, `nothrottle=true`, `noipchecks=true`. |
| **poke-env** | [hsahovic/poke-env](https://github.com/hsahovic/poke-env) (`v0.17.0`, Nov 2026) | Actively maintained | **MIT** | **Reuse directly.** Note breaking change vs old tutorials: `poke_env.environment` moved to `poke_env.battle`; `battle.valid_orders` is built-in. |
| **PokéLLMon** (2024) | [git-disl/PokeLLMon](https://github.com/git-disl/PokeLLMon), [arXiv:2402.01118](https://arxiv.org/abs/2402.01118) | Stale (last push Oct 2024) | **No LICENSE file** (All Rights Reserved — do not copy code) | Showed zero-shot GPT-4 hallucinates mechanics & "panic-switches"; injecting type/move knowledge helps immediately. |
| **PokéChamp** (ICML 2025 Spotlight) | [sethkarten/pokechamp](https://github.com/sethkarten/pokechamp), [arXiv:2503.04094](https://arxiv.org/abs/2503.04094) | Active (pushed Nov 2026; MCP + VGC added) | **Modified MIT prohibiting use/distribution in Japan** (not OSI-approved) | Zero-shot LLM + minimax search + Bayesian set predictor (~1300–1500 projected Elo). Reuse ideas & HF datasets ([`milkkarten/pokechamp`](https://huggingface.co/datasets/milkkarten/pokechamp), [`milkkarten/pokemon-showdown-replays-merged`](https://huggingface.co/datasets/milkkarten/pokemon-showdown-replays-merged)); respect license restrictions on code. |
| **Metamon** (UT Austin, RLC 2025) | [UT-Austin-RPL/metamon](https://github.com/UT-Austin-RPL/metamon), [arXiv:2504.04395](https://arxiv.org/abs/2504.04395) | Active (pushed Jul 2026; 40+ checkpoints on HF) | **MIT** | Specialist non-LLM Transformer (15M–200M) trained via offline RL + diverse self-play (5M human + 20M synthetic trajectories). `Kakuna` hits 71% GXE in Gen 9 OU; `TaurosEnsemble` hit #1 Gen 1 OU ladder. **OU only (no Randbats)**, but MIT replay reconstruction code & lessons (diverse opponents prevent self-play collapse; two-hot value classification) are gold. |
| **The PokéAgent Challenge** (NeurIPS 2025) | [arXiv:2603.15563](https://arxiv.org/abs/2603.15563), [pokeagentchallenge.com](https://pokeagentchallenge.com) | Living benchmark (Mar 2026 paper, live leaderboard) | Paper CC BY-NC-SA 4.0; infra open | 100+ teams. **Key result:** specialist RL (Metamon) & search (Foul Play) dominated prompted/harness LLMs; Pokémon battling is nearly orthogonal to standard LLM benchmarks (rank-2 BenchPress SVD captures only 27% of GXE variance). |
| **Foul Play** | [pmariglia/foul-play](https://github.com/pmariglia/foul-play) | Active (pushed Oct 2026) | **GPL-3.0** (copyleft) | Rust `poke-engine` + root-parallelized MCTS/expectimax. Won #1 Gen 9 OU in PokéAgent Challenge; ~1800–2000+ Elo ceiling on `gen9randombattle`. **Reuse as a separate-process eval ceiling opponent** over local Showdown (keeps GPL isolated). |
| **Porimon** (FLLM 2026) | [arXiv:2609.32544](https://arxiv.org/abs/2609.32544) (Oct 2026) | Recent paper | Paper CC BY 4.0 | Extended PokéLLMon with Smogon Damage Calc API + long/short-term KAG. **Key ablation:** exact damage/item/ability features drove almost all improvement; long-term memory was inconclusive. |
| **Community Pure-RL Bots** | [Poke-bot](https://github.com/dheerajkumar2005/Poke-bot), [RotomAI](https://github.com/BhaveshThapar/RotomAI) | 2025–2026 repos | Check per repo | `Poke-bot` hit 1900+ Randbats Elo with a compact RL Transformer (>250M PPO steps on 3090). `RotomAI` is a crucial warning: scored 73% vs `SimpleHeuristicsPlayer` yet only 1004 Elo / 30% GXE against humans — heuristic win rate alone is not enough! |

---

## 3. Recommended Architecture

### Decision: **Small Base LLM Policy + Learned Value Head (Actor-Critic) with Direct Single-Token Actions + Precomputed Domain Hints**

Specifically: **`Qwen/Qwen3-1.7B-Base` (LoRA) reading our compact text state (enriched with deterministic type multipliers, STAB, speed stats, and 3-turn history) → outputting a masked categorical distribution over 13 canonical single-token action IDs + a learned Value Head $V_\phi(s_t)$ on the last token's hidden state.**

### Why This Beats the Alternatives on 12GB + 1–3h Runs
1. **Direct-Action vs. Chain-of-Thought (GRPO):**
   - **Throughput:** Direct-action runs **30–50× faster** (~35K battles/hour vs ~1K battles/hour), which is the difference between a real RL signal and pure noise inside a 2-hour weekend run.
   - **Credit assignment:** GRPO relies on sampling $G$ completions *from the exact same prompt*. In a 30-turn interactive Showdown battle, you cannot cheaply fork mid-battle server states for $G=8$ rollouts; trajectory-level GRPO across different random battles suffers massive variance from random team luck. A **learned Value Head $V_\phi(s_t)$** directly predicts expected win probability from *this team matchup and board state*, slashing advantage variance via GAE!
   - **Zero-error legality:** Masking logits over the 13 canonical single-token IDs guarantees 100.0% legal moves and exact $\log \pi_\theta(a \mid s)$ / entropy $H(\pi)$ in one prefill pass.
2. **"LLM as Reasoner" vs. "LLM as a Text-Reading Policy Network":**
   - Be explicit: this architecture treats the LLM as a **pretrained text-reading policy+value network**, not a verbose verbal reasoner.
   - What pretraining buys you is **sample efficiency**: `Qwen3-1.7B-Base` already encodes Pokémon species, abilities (`Unaware`, `Good as Gold`, `Levitate`), items (`Choice Scarf`, `Heavy-Duty Boots`), and move semantics from web pretraining, so it learns strong play from 150K–300K turns instead of 20M+.
   - **Will a specialized non-LLM RL or search bot beat this LLM?** **Yes, at the ceiling.** `Foul Play` (Rust MCTS with exact damage engine) and `Metamon` (142M custom transformer trained on 25M trajectories) or a scratch 10M RL Transformer run at 100× higher env throughput. You will know exactly where your LLM stands by benchmarking against **`SimpleHeuristicsPlayer`** (low bar), **Stage 0 prompted LLM** (zero-shot LLM bar), and **`Foul Play`** (expert search ceiling).
3. **Future Escape Hatch:** Because you already have a learned Value Head $V_\phi(s_t)$, if pure 1-pass LLM play plateaus around 1450–1600 Elo, you can cleanly add a **1-ply damage/expectimax reranker** in Phase 2 without redesigning the model!

---

## 4. State Representation & Action Output Format

Implemented in [`src/alphabattle/state.py`](../src/alphabattle/state.py).

### 4.1 Canonical Single-Token Action IDs (13 Symbols)
Every legal order maps to a **slot-stable single ASCII character** (guaranteed 1 token in Qwen3 / Llama 3 / Gemma):
- `1`, `2`, `3`, `4` — Use `available_moves[0..3]`
- `A`, `B`, `C`, `D` — Terastallize + `available_moves[0..3]` (only present when `battle.can_tera` is True)
- `5`, `6`, `7`, `8`, `9` — Switch to `available_switches[0..4]` (or `REVIVE` under Gen 9 Revival Blessing)

Forced single actions (`len(legal_actions) == 1`) short-circuit immediately in Python without wasting an LLM call.

### 4.2 Compact Text State (~350–550 Tokens)
- **Header:** format, turn, my/opp Tera availability, flags (`FORCE_SWITCH`, `TRAPPED`, `REVIVING`).
- **Field (non-empty only):** weather, terrain/room, my hazards/screens, opp hazards/screens.
- **Active Pokémon (Ours & Opponent):** species, level (critical in Randbats!), effective types & Tera type, HP (`cur/max (%)` for ours, `%` for opp), status, non-zero stat boosts, item, ability (or possible abilities), volatiles, speed (`spe` for ours, `base_spe` for opp), and opponent's revealed moves.
- **Benches:** compact 1-liner per bench mon (`FNT` if fainted) + `unseen=k` count for opponent.
- **Short History:** last 3 turns (`T6: chose [3] MOVE swordsdance -> opp(...) last=...`).
- **Legal Actions Block:** lists ONLY legal IDs with precomputed deterministic facts — move type/category/BP/Acc/PP, **exact type effectiveness (`eff x0..x4`)**, **STAB / Tera STAB**, priority/heal/pivot/boost/status tags, and defensive type matchups (`takes:...`) on switches.

### 4.3 Unambiguous Output Constraining
- **Stage 0 (`llama-server` / Ollama):** Strict `json_schema` `enum` over current `legal_ids`, or `llama-server` 1-token GBNF grammar `root ::= ("1" | "2" | "A" | "5")`.
- **Stage 1–2 (PyTorch):** Gather logits at the exact vocabulary IDs of `legal_ids`, mask illegal entries to $-\infty$, sample from `Categorical(logits)`. **0% parse errors by construction.**

---

## 5. Staged Training Pipeline (12GB RTX 3080, Native Windows, 1–3h Runs)

### Stage 0: Prompted Off-the-Shelf Baseline (Milestone 1)
- Serve `Qwen3-4B` (Q4_K_M, ~2.6 GB VRAM) and `Qwen3-1.7B` (Q4_K_M, ~1.1 GB VRAM) locally via `llama-server.exe` or Ollama.
- Evaluate both `direct` mode (`constrain=gbnf` or `json_schema`) and `cot` mode (`constrain=json_schema`) on 100 games each vs `RandomPlayer`, `MaxBasePowerPlayer`, and `SimpleHeuristicsPlayer`.
- This establishes the exact zero-shot floor to beat.

### Stage 1: SFT / Behavioral Cloning on Human Replays (Milestones 2–3)
1. **Milestone 2 Smoke Test (Zero Sim2Sim Gap):** Collect 5,000 local games of `SimpleHeuristicsPlayer` using live `poke-env` states (~150K turns, zero reconstruction noise). Train LoRA-SFT on `Qwen3-1.7B-Base` (1h) and verify ≥85% validation top-1 match and ~50% head-to-head vs `SimpleHeuristicsPlayer`. This proves the training + inference pipeline bug-free before touching human replays.
2. **Human Replay Data Sources (`gen9randombattle`):**
   - [`milkkarten/pokemon-showdown-replays-merged`](https://huggingface.co/datasets/milkkarten/pokemon-showdown-replays-merged) (29.1M total replays; **703,345 `[Gen 9] RANDOMBATTLE` replays** with `rating`, `turns`, `log` — stream with `datasets.load_dataset(..., streaming=True)`).
   - Secondary: [`milkkarten/pokechamp`](https://huggingface.co/datasets/milkkarten/pokechamp) (2.13M battles, 2024–2025) and [`HolidayOugi/pokemon-showdown-replays`](https://huggingface.co/datasets/HolidayOugi/pokemon-showdown-replays).
3. **Spectator-to-POV Reconstruction (Solving the Sim2Sim Gap):**
   - Public replays are spectator logs (missing private `|request|` packets and unrevealed bench sets).
   - **Two-pass parser:** Pass 1 scans the full replay to collect each player's *actual* species, levels, Tera types, and revealed moves/items/abilities (optionally filling unrevealed own-team move slots from Showdown's official `data/random-battles/gen9/sets.json`). Pass 2 replays protocol events causally through `poke-env`'s `Battle.parse_message`, keeping the **opponent's unrevealed info strictly masked** while matching the acting player's chosen move/switch/Tera to our canonical Action IDs (`1-4`, `A-D`, `5-9`).
   - Conservatively drop games with `turns < 8`, forfeits before turn 10, Zoroark/Illusion tracking mismatches, or unmapped actions.
4. **Filtering & Dataset Targets:**
   - Filter `rating >= 1550` (try `>= 1600` first; keep **winner's POV** to raise effective skill without starving sample count).
   - Target **15,000–25,000 filtered battles → 150K–300K turns** for a 1.5–2.5h weekend run. Split train/val strictly by **`battle_id`** (hold out 2,000 battles) to prevent leakage!
5. **Exact 12GB Windows SFT Hyperparameters (`Qwen/Qwen3-1.7B-Base`):**
   - Stack: `torch` (CUDA 12.4/12.6), `transformers`, `peft`, `trl` (`SFTTrainer`), `bitsandbytes` (`paged_adamw_8bit`), `attn_implementation="sdpa"`.
   - LoRA: `bf16` base (no 4-bit needed at 1.7B!), `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `target_modules="all-linear"`.
   - Batch & schedule: `max_length=1024`, `per_device_train_batch_size=8`, `gradient_accumulation_steps=8` (eff. batch 64), `gradient_checkpointing=True`, `lr=2e-4`, cosine decay, 3% warmup, 1–2 epochs.
   - **Critical:** Set **completion-only loss** (mask all state prompt tokens to `-100`) — or better yet, compute **masked cross-entropy strictly over the current turn's `legal_token_ids`** so every gradient bit directly improves legal move selection!
   - Expected peak VRAM: **~7.5–8.5 GB** (<10.5 GB Windows ceiling). Runtime: **~1.5–2.5 hours**.

### Stage 1.5: Value Head Pretraining (Milestone 4)
- Attach a lightweight 2-layer MLP Critic head `hidden_dim -> 256 -> 1` (tanh win value in `[-1, +1]`, or Metamon-style 51-bin two-hot cross-entropy!) on top of the last prompt token's hidden representation.
- Freeze the LoRA policy (or use a tiny LR) and train the Value Head for **30–45 minutes** to predict battle outcomes (`+1` win / `-1` loss) from your reconstructed replay trajectories.
- **Why this matters:** A randomly initialized value head produces noisy advantage estimates that corrupt a good SFT policy during the first 50 PPO steps. Warm-starting the critic makes Stage 2 PPO dramatically more stable.

### Stage 2: Online Masked-Action PPO on Local Showdown (Milestones 5–6)
1. **Parallel Rollout Engine (Native Windows):**
   - Run **2 local Pokémon Showdown Node instances** (ports `8000` & `8001` via `scripts/start_showdown.ps1 -Instances 2`) to utilize the i7-10700K smoothly.
   - Run **16–24 concurrent `poke-env` battles** feeding an `asyncio.Queue` micro-batcher (`batch_size=16`, `max_wait=15ms`) that performs **one batched `bf16` forward pass** on `Qwen3-1.7B-Base + LoRA + Value Head`, returning masked action samples, `log_prob`, and `V(s_t)`.
   - Collect `4,096` transitions (~130 battles, ~20 seconds) → run 3 PPO epochs (`clip=0.2`, `GAE gamma=0.99, lam=0.95`, `micro_batch=8`, `grad_accum=8`, `lr=2e-5`, ~25 seconds) → repeat!
   - **Per 2-hour weekend session:** ~150 PPO updates = **~600K transitions (~20,000 full battles)**.
2. **KL Anchor (Zero Extra Base VRAM):**
   - Add per-turn KL penalty $-\beta \, D_{\text{KL}}(\pi_\theta(\cdot|s) \| \pi_{\text{SFT}}(\cdot|s))$ with $\beta \in [0.02, 0.05]$.
   - Compute $\pi_{\text{SFT}}$ without loading a second 1.7B model by holding a second PEFT adapter (`"sft_ref"`, ~40 MB) on the same base weights and switching active adapters with `model.set_adapter("sft_ref")` under `torch.no_grad()`!
3. **Reward Function & Reward Hacking Guardrails:**
   - **Primary reward:** `+1.0` win, `-1.0` loss, `0.0` tie.
   - **Optional potential-based shaping (annealed to 0 over training):**
     $$\Phi(s) = 0.05 \cdot (\text{opp\_fainted} - \text{my\_fainted}) + 0.02 \cdot \left(\sum \text{my\_hp\_frac} - \sum \text{opp\_hp\_frac}\right)$$
     Step shaping reward $r_{\text{shape}}(s, s') = \Phi(s') - \Phi(s)$.
   - **Guardrail #1 (No stall/panic-switch hacking):** Because $r_{\text{shape}}$ is strictly potential-based ($\sum_t (\Phi(s_{t+1}) - \Phi(s_t)) = \Phi(s_T) - \Phi(s_0)$), healing/switching loops sum to **zero**! Total shaping magnitude is capped well below `1.0` so winning always dominates.
   - **Guardrail #2 (No self-play collapse — Metamon's key lesson):** Never train 100% against your latest checkpoint. Maintain a **mixed opponent pool**: 40% latest checkpoint, 30% past checkpoints (snapshot every 25 updates), 20% `SimpleHeuristicsPlayer` / `MaxBasePowerPlayer`, 10% frozen SFT anchor.
   - **Guardrail #3 (Entropy & KL monitoring):** Watch mean policy entropy (`0.3–1.2 nats` healthy; `<0.1` means premature collapse → raise entropy coef `0.005->0.015`) and KL divergence (`>0.15` → raise $\beta$ or lower LR).

---

## 6. Evaluation Harness

Implemented in [`scripts/eval.py`](../scripts/eval.py) and [`src/alphabattle/metrics.py`](../src/alphabattle/metrics.py).

- **Fixed opponents:** `RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`, plus head-to-head against prior checkpoints (`Stage 0`, `SFT-v1`, `RL-v1`, ...) and optionally `Foul Play` (separate local process).
- **Statistics:** Wilson score 95% CIs (accurate near 0% and 100%), two-proportion z-test $p$-values for checkpoint comparisons, approximate logistic $\Delta\text{Elo}$.
- **Sample sizes:** $N = 100$ games/opponent for rapid weekend checks ($\pm 9.8\%$ 95% CI at 50% WR); $N = 200\text{–}400$ for milestone sign-off ($\pm 5\text{–}7\%$ CI).
- **Behavioral diagnostics tracked every run:** mean turns/battle, **fallback/parse-error rate** (target `0.00%`), mean & p95 latency (ms), move / Tera / switch rates (catches panic-switching immediately!).
- **Public Showdown Ladder protocol (Milestone 6 only):** Pre-register $N = 100$ rated games on a dedicated account, respect Showdown bot etiquette (no chat spam, normal pacing, no forfeit farming), and report official Elo + GXE + Glicko-1 honestly alongside negative results.

---

## 7. Weekend Milestone Plan

| Weekend | Milestone | Compute / Time | Clear "Done" Criterion |
|---|---|---|---|
| **Weekend 1** | **Local Showdown + Eval Harness + Stage 0 Baseline** | Local inference (~30–60 min) | Local Showdown runs on Windows; `scripts/eval.py` completes 100 games each vs `Random`, `MaxBasePower`, `SimpleHeuristics` for `Qwen3-4B` (direct vs CoT) & `Qwen3-1.7B` with `0.00%` fallback rate and saved JSON report. |
| **Weekend 2** | **Heuristic Distillation (Zero-Gap SFT Smoke Test)** | 1–1.5h SFT on 3080 | Record 5K local games of `SimpleHeuristicsPlayer`; train `Qwen3-1.7B-Base` bf16 LoRA; hit ≥85% held-out top-1 action match, ~50%±10% vs `SimpleHeuristicsPlayer`, and >100 decisions/s in direct PyTorch masked-logit mode. |
| **Weekend 3–4** | **Human Replay Parser + High-Elo SFT (Stage 1)** | 2–2.5h SFT on 3080 | Stream `gen9randombattle` (`rating >= 1550`) from `milkkarten/pokemon-showdown-replays-merged`; reconstruct 150K–300K POV turns (battle-ID train/val split); LoRA-SFT `Qwen3-1.7B-Base`; **beat `SimpleHeuristicsPlayer` >60–70% over 200 games (95% CI lower bound >55%)** and clearly beat Stage 0. |
| **Weekend 5** | **Value Head Pretraining (Stage 1.5)** | 30–45 min on 3080 | Train Critic head on replay outcomes; verify validation win-probability ROC-AUC >0.75 mid-game and clean calibration curve before starting PPO. |
| **Weekend 6–7** | **Parallel Showdown PPO + Mixed Pool (Stage 2)** | Two 2–3h RL runs on 3080 | Batched async Showdown loop runs ≥8 battles/s; KL-anchored PPO vs mixed pool beats Stage 1 SFT checkpoint >58–62% (200 games, 95% CI >50%) AND beats `SimpleHeuristicsPlayer` >80%. |
| **Weekend 8** | **League RL Polish + Ladder & Foul Play Reality Check** | 2h RL + 100 ladder games | Pre-registered 100-game public `gen9randombattle` ladder run (target 1400–1600+ Elo, GXE >55–65%) + 100-game local match vs `Foul Play` to quantify the LLM-vs-search gap. |
| **Phase 2 (Opt.)** | **Gen 9 OU (Fixed Teams) or 1-Ply Value Reranker** | 2–3 weekends | Add Team Preview prompt block + fixed sample teams (from `UT-Austin-RPL/metamon`), benchmark head-to-head against Metamon's `Kakuna`/`Kadabra` baselines. |
