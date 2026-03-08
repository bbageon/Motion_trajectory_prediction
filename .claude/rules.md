# Motion Trajectory Prediction - Claude Code Rules (Compact)

## A. Goal
- Primary goal: produce reproducible, quantitative evidence for LLM-based motion trajectory prediction.
- Priority order: `data integrity > fair baseline comparison > model gains > convenience`.

## B. Project Map (only what matters)
- `BaseLine/01_offlineDownload.py`: download base model snapshot.
- `BaseLine/03_TokenizerExtend.py`: add structured numeric tokens and save extended model.
- `BaseLine/04_mp4ToJson.py`: MP4 -> `pose_sequence` JSON.
- `BaseLine/05_delta_convertToFinetuning2.py`: build delta JSONL splits (`train/val/test`).
- `BaseLine/05_delta_convertToFinetuning_fileLevelSplit.py`: delta file-level split JSONL (`train/val/test`).
- `BaseLine/05_abs_covertToFinetuning2.py`: build absolute JSONL splits (`train/val/test`).
- `BaseLine/05_abs_convertToFinetuning_fileLevelSplit.py`: absolute file-level split JSONL (`train/val/test`).
- `BaseLine/06_finetune_llama2_lora.py`: LoRA finetuning entrypoint.
- `delta/01_test2_delta.py`: single-sample inference/visualization (`jsonl` or `pose` mode).
- `delta/`, `absolute/`: inference + parsing + visualization scripts.

## B1. Current Training Structure (must match code)
- Base model path default: `../Meta-Llama-3.1-8B_tokenizerExtension`.
- Prompt format in training is direct dataset prompt (no `### Instruction ###` wrapper).
- Data construction window policy: `OBS=10`, `PRED=8`, `STRIDE=3` for both delta/absolute converters.
- LoRA target modules are fixed to attention projections:
  - `q_proj`, `k_proj`, `v_proj`, `o_proj`
- Token I/O exception rule:
  - use `--token-io-mode` (`full`, `special_only`, `lm_head_only`, `lm_head_special_only`, `off`)
  - current recommended default for stability: `lm_head_special_only`
  - optional full freeze: `--no-train-token-io` (equivalent to `off`)
- Precision policy:
  - model load: `torch_dtype=torch.bfloat16` (RTX 5090 target)
  - trainer precision: `bf16=True` on CUDA
- Data split policy:
  - default training track uses sample-level JSONL from:
    - delta: `05_delta_convertToFinetuning2.py`
    - absolute: `05_abs_covertToFinetuning2.py`
  - train with `*_train.jsonl`
  - do not train on `*_val.jsonl`, `*_test.jsonl`
- Checkpoint selection policy:
  - if validation is enabled, select best checkpoint by minimum `eval_loss`
  - if validation is disabled/skipped, treat last checkpoint as training output

## C. Non-negotiable Experiment Rules
- Keep split policy fixed at `8:1:1` unless explicitly changed and documented.
- Train only with `*_train.jsonl`; never train on `val/test`.
- Maintain two split tracks:
  - sample-level split for continuity with prior runs
  - file-level split for leakage-safe reporting
- Keep delta/absolute data pipelines symmetric:
  - both must provide sample-level + file-level split scripts
  - both must use same window policy (`OBS=10`, `PRED=8`, `STRIDE=3`)
- Compare at least:
  1. Delta-token LLM
  2. Absolute-token LLM
  3. One non-LLM baseline (recommended LSTM)
- Use same split, horizon, and metrics across all methods.

## D. Metrics & Error Analysis (required)
- Report at minimum:
  - MPJPE or coordinate MAE
  - Horizon-wise error curve (drift)
  - Joint-wise error breakdown (e.g., hip/ankle/wrist)
- Analyze failure modes:
  - long-horizon drift
  - action-specific failures
  - output format/parsing failure rate

## D1. Inference Error Evaluation Protocol (required)
- Mid-training check:
  - monitor `eval_loss` on validation only (not final quality claim).
- Model selection:
  - choose checkpoint by best validation `eval_loss` (not by last step)
  - final report model must be the selected best checkpoint
- Final evaluation must be done on `*_test.jsonl` with actual generation output.
- Compute errors after decoding outputs to coordinates:
  - delta model: decode `(dx, dy)` then accumulate to absolute coordinates
  - absolute model: decode as absolute coordinates directly
- Report all of the following together:
  - overall MAE/MPJPE
  - horizon-wise error (`t+1 ... t+H`) to show drift growth
  - action-wise error (e.g., armRaise vs armRaise90 vs armVertical)
  - parse success rate (% outputs that can be fully parsed)
- In `delta/01_test2_delta.py` pose-mode inference:
  - prompt must be aligned to training format (`Observed motion deltas: ...`)
  - append `Next motion deltas:` before generation start

## D2. Current Evidence Snapshot (2026-03-08)
- Delta test report (`delta/delta_eval_report.json`) currently shows:
  - `parse_success_rate=1.0`, `aligned_success_rate=1.0`
  - `overall_mean_dist=0.0045307`
- Do not over-claim from sample-level split only.
- Paper-quality claims must include file-level split results to reduce leakage risk.

## E. Runtime Safety Checklist
- Before training:
  - confirm CUDA (`BaseLine/02_Cuda_available_test.py` or `torch.cuda.is_available()`).
  - confirm model path, tokenizer-extension path, train JSONL path exist.
- Prefer path resolution relative to script location (`__file__`), not shell cwd.
- Keep CLI overrides for model/data/output paths.

## F. Reproducibility Checklist
- Fix and log random seed.
- Log: model id/path, dataset split files, key hyperparameters, git commit hash, run date.
- For report-quality claims, rerun key experiments and record variance.

## G. Git/Data Hygiene
- Do not commit large artifacts: base model weights, checkpoints, raw videos, generated media.
- Keep commits atomic and message intent clearly (`feat`, `fix`, `exp`, `eval`).
- If behavior changes, update README or experiment note in same PR/commit series.

## H. Fast Operating Procedure
1. GPU check
2. Model download
3. Tokenizer extension
4. Dataset conversion/splitting
5. LoRA train on train split
6. Evaluate on val/test and produce drift/error analysis

## I. Why These Rules (evidence)
- LoRA efficiency for adaptation: Hu et al., 2021, https://arxiv.org/abs/2106.09685
- Compute/data discipline (scaling tradeoffs): Hoffmann et al. (Chinchilla), 2022, https://arxiv.org/abs/2203.15556
- Reproducibility standards in ML: Pineau et al., 2021, https://jmlr.org/papers/v22/20-303.html
- Leakage-aware split discipline (industry): Google "Rules of ML", https://developers.google.com/machine-learning/guides/rules-of-ml
