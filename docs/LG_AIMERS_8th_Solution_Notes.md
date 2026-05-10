# LG Aimers 8th — On-Device EXAONE Model Compression (2nd Place) — Technical Notes

This document summarizes the **LG Aimers 8th** hackathon track (*model lightweighting for on-device AI with EXAONE*) and our competition pipeline: preference optimization (**DPO**), **GPTQ / llmcompressor** quantization schemes (**FP8**, **W8A8**, **W4A16**), **calibration-data design**, and offline benchmarking with **vLLM**.  
Artifacts live under local workspaces: `sehun/lg aimers/` (submission notebook) and `aimers/lg_contest/` (training, quantization, benchmarks).

---

## Official sources & press

| Resource | URL |
|----------|-----|
| LG Aimers (program homepage) | [https://www.lgaimers.ai/](https://www.lgaimers.ai/) |
| DACON — LG Aimers 8th competition hub | [https://dacon.io/competitions/official/236673/overview/description](https://dacon.io/competitions/official/236673/overview/description) |
| LG press release (KR) | [https://www.lg.co.kr/media/release/29342](https://www.lg.co.kr/media/release/29342) |
| AI Times article — LG on-device lightweighting hackathon (KR) | [https://www.aitimes.com/news/articleView.html?idxno=208841](https://www.aitimes.com/news/articleView.html?idxno=208841) |
| EXAONE 4.0 technical report (arXiv) | [https://arxiv.org/abs/2507.11407](https://arxiv.org/abs/2507.11407) |

---

## Task framing

- **Goal**: compress **EXAONE 4.0 ~1.2B** for **on-device** constraints while preserving accuracy on Korean / multilingual benchmarks (contest suite aligns with **KMMLU**, **MMLU-Redux**, reasoning tracks, etc.).
- **Stack**: Hugging Face **Transformers**, **PEFT LoRA**, **TRL `DPOTrainer`**, **vllm** for throughput-aware evaluation, **llmcompressor** (`GPTQModifier`, `oneshot`) for post-training quantization.

---

## DPO (Direct Preference Optimization)

### Implementation

- **Script**: `aimers/lg_contest/scripts/train_dpo_lora.py`
- **Trainer**: Hugging Face **TRL** `DPOTrainer` + `DPOConfig`; **reference model** frozen; **policy** = base Causal LM + **LoRA**.
- **LoRA** (from saved adapter config): **rank `r = 64`**, **`lora_alpha = 128`**, **dropout `0`**, targets `{q,k,v,o}_proj`, `gate_proj`, `up_proj`, `down_proj`.
- **Precision**: **bf16** on trainable path. **FP8/INT8 quantized checkpoints are not used for DPO backward** (Float8 weight backward not supported); training uses a **bf16/fp16 base** then merges / re-quantizes in a separate stage — see `DPO_METHODOLOGY.md`.

### Data

- Pairwise JSONL: `prompt` / `chosen` / `rejected`, optional `prompt_format: "messages_json"` for EXAONE chat templates.
- Provenance includes **Ultrafeedback (binarized)**, **HH-RLHF-style** sources, and append scripts under `scripts/append_ultrafeedback_dpo.py` and `download_recovery_datasets.py`.
- `apply_chat_template` with **`enable_thinking=False`** where supported, for consistent train/infer formatting.

### Observed run statistics (artifact: `dpo_training_report.json`)

- **Global steps**: **781** (single-epoch schedule in that run).
- **DPO loss** (representative): started about **0.68** (step 10), trended to about **0.59** by the last logged step (~**0.592** at step 780).
- **Implicit preference accuracy** (`rewards/accuracies` in TRL logs): moved from the **~0.44–0.64** band early on to **~0.65–0.76** in mid/late training (e.g. **0.7625** at step 620) — useful as a *relative* monitor, not a substitute for task accuracy.
- **Artifacts**: `plots/dpo_loss_lr.png`, `dpo_rlhf_metrics.png`, `dpo_all_numeric.png`, `dpo_grad_norm.png`.

### Hyperparameters (defaults in `docs/DPO_METHODOLOGY.md` / CLI)

- **learning_rate** `5e-6`, **beta** `0.1`, **per_device_train_batch_size** `1`, **gradient_accumulation_steps** `8`, **max_length** `2048`, **max_prompt_length** `1024` when the TRL version accepts it.

---

## Quantization experiments

### A. Final competition-style path — **FP8 GPTQ** (`gptq_final`)

- **Tooling**: `llmcompressor` `oneshot` + `GPTQModifier`.
- **Saved recipe** (`data/models/gptq_final/recipe.yaml`): **`scheme: FP8`**, `targets: [Linear]`, `ignore: [embed_tokens, lm_head]`, **block_size 128**, **dampening_frac 0.01**, `actorder: static`.
- **Notebook** `sehun/lg aimers/final_solution.ipynb` documents the full **calibration** design for this run.

### B. **W8A8** (Int8 weight / int8 activations) vs **FP8** on **Blackwell (SM120+)**

- **vLLM** on **SM120+** may **not** execute some **W8A8 Int8** checkpoints (missing Cutlass INT8 path). The benchmark harness (`scripts/benchmark.py`) implements **`sm120_w8a8_auto_substitute`**: if Int8 W8A8 load fails, it **falls back to an FP8** checkpoint (e.g. under `data/models/FP8_DYNAMIC` or user override).
- **Practical lesson**: for the same *numerical* recipe, **deployment target GPU** must be co-designed with the **quant format** (W8A8 vs FP8).

### C. **W4A16** ablations

- Baseline notebooks (e.g. `scripts2/baseline.ipynb`) use **`SCHEME = "W4A16"`**, **`NUM_CALIBRATION_SAMPLES = 256`**, **`MAX_SEQUENCE_LENGTH = 512`** — a lighter calibration budget useful for fast iteration before scaling to **512×2048** runs.

### “Pruning” naming

- Several benchmark output directories use the suffix **`_pruned`** (e.g. `output/W8A8_DYNAMIC_pruned`). **Structured pruning** was **not** enabled in the sampled llmcompressor logs (`skip_sparsity_compression_stats` / no sparsity compressor). Treat **`pruned` here as a run label / legacy folder name**, not explicit unstructured pruning unless you add a sparsity modifier.

---

## Calibration data engineering (512 × 2048 in `final_solution`)

**Total**: **`NUM_CALIBRATION_SAMPLES = 512`**, **`MAX_SEQUENCE_LENGTH = 2048`**, **`SEED = 42`**.

**Per-source quotas** (must sum to 512):

| Bucket | Count | Role |
|--------|------:|------|
| `manta_plain` | 130 | LGAI **MANTA-1M**, non-reasoning chat template |
| `manta_reasoning` | 90 | Same pool, **`enable_thinking=True`** |
| `gsm_plain` / `gsm_reasoning` | 48 + 40 | **GSM8K** math, plain vs chain-of-thought style |
| `kmmlu` | 124 | **KMMLU-Redux** multiple-choice, EXAONE chat format |
| `teleia_es` | 32 | Spanish **TELEIA**; **FLORES `spa_Latn` backfill** if short |
| `wikitext_long` | 32 | **WikiText-103** long contexts; **EN vs KO user wrap** on alternating rows |
| `agent_tools` | 16 | Synthetic **function-calling** style prompts (tool schema) |

**Filters & heuristics**

- **MANTA** complexity: start at **`MANTA_MIN_COMPLEXITY = 8`**; if the pool is too small, relax one notch (**≥ 7**).
- Drop texts below **`MIN_CALIB_TOKENS = 48`** tokens (tokenizer-aware).
- **WikiText** chunks capped by **`WIKI_MAX_CHARS = 5500`**.
- If filtering removes too many samples, **pad from unused MANTA pool rows**.

---

## Offline benchmark harness (`scripts/benchmark.py`)

- Measures **per-subject accuracy**, **invalid rate**, **tokens/sec**, wall time, token-length histograms — for **contest benchmarks** (`KMMLU`, `MMLU-Redux`, …) plus optional auxiliary sets.
- **`RUN_CONFIG` defaults**: merged **PEFT → single checkpoint** for fair latency (`merge_peft_lora=True`), **`max_model_len` 16384**, optional **`ngram_spec`** speculative decoding, **`auto_fp8_fallback`** for SM120 Int8 issues.
- Example CSV output on disk: `output/W8A8_DYNAMIC2.csv` (row-level breakdown + totals such as **ARC-Challenge**, **KoBEST** splits, **GSM8K** aggregate).

---

## Repository map (local)

| Path | Contents |
|------|----------|
| `sehun/lg aimers/final_solution.ipynb` | End-to-end **FP8 GPTQ** + calibration story for final packaging |
| `aimers/lg_contest/docs/DPO_METHODOLOGY.md` | Math + code mapping for DPO |
| `aimers/lg_contest/scripts/train_dpo_lora.py` | DPO + LoRA entrypoint |
| `aimers/lg_contest/data/recovery_checkpoints/dpo_lora/` | LoRA adapter + `dpo_training_report.json` |
| `aimers/lg_contest/data/models/gptq_final/` | FP8 GPTQ export + `recipe.yaml` |
| `aimers/lg_contest/scripts/benchmark.py` | vLLM batch evaluator |

---

## Repro / environment notes

- Match **CUDA / PyTorch / vLLM / llmcompressor** versions to avoid kernel gaps (especially **FP8** on **SM120** and **Int8** fallbacks).
- For long-context eval, watch **GPU memory** vs `max_model_len`.

---

## Disclaimer

Rankings and leaderboard positions refer to the **LG Aimers / DACON** contest instance you participated in; always cite **official DACON / organizer** results for awards or résumé claims.

---

*Last updated: 2026-05-10 — synced from private experiment repos for documentation.*
