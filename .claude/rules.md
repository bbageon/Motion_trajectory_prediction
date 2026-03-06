# Motion Trajectory Prediction - Claude Code Rules (Compact)

## A. Goal
- Primary goal: produce reproducible, quantitative evidence for LLM-based motion trajectory prediction.
- Priority order: `data integrity > fair baseline comparison > model gains > convenience`.

## B. Project Map (only what matters)
- `BaseLine/01_offlineDownload.py`: download base model snapshot.
- `BaseLine/03_TokenizerExtend.py`: add structured numeric tokens and save extended model.
- `BaseLine/04_mp4ToJson.py`: MP4 -> `pose_sequence` JSON.
- `BaseLine/05_delta_convertToFinetuning2.py`: build delta JSONL splits (`train/val/test`).
- `BaseLine/05_abs_covertToFinetuning2.py`: build absolute-coordinate JSONL.
- `BaseLine/06_finetune_llama2_lora.py`: LoRA finetuning entrypoint.
- `delta/`, `absolute/`: inference + parsing + visualization scripts.

## C. Non-negotiable Experiment Rules
- Keep split policy fixed at `8:1:1` unless explicitly changed and documented.
- Train only with `*_train.jsonl`; never train on `val/test`.
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
