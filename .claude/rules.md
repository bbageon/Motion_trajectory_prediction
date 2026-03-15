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

## B2. max-length Setting Rule (critical)
- All samples (delta & absolute fileLevel) are exactly fixed-length after tokenization:
  - delta fileLevel: prompt ~1029 + completion ~815 + EOS = ~1845 tokens
  - absolute fileLevel: prompt ~1120 + completion ~904 + EOS = 2025 tokens (all samples identical)
- **`--max-length` must be set to 4096** for training (not default 1024).
  - Reason: inference context = input(~1120) + output(~2048) = ~3168 tokens; position embedding must cover this.
  - With max-length=1024, absolute prompt is truncated to ~119 tokens (only ~1 frame of context) → training failure.
- Training command reference (absolute fileLevel, correct):
  ```
  python 06_finetune_llama2_lora.py \
    --train-jsonl finetune_dataset_absolute_fileLevel_train.jsonl \
    --val-jsonl   finetune_dataset_absolute_fileLevel_val.jsonl \
    --output-dir  motionQaAbsoluteFileLevelV2 \
    --max-length 4096 --learning-rate 5e-5 --epochs 6 \
    --batch-size 1 --grad-accum 8 \
    --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 \
    --token-io-mode lm_head_special_only
  ```

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
- In `delta/01_test2_delta_evalation.py` JSONL evaluation:
  - prefer `--append-target-prefix` because dataset `prompt` is `Observed motion deltas: ...`
  - target marker `Next motion deltas:` lives at the start of `completion`, so appending it at inference improves format alignment

## D2. Current Evidence Snapshot (2026-03-10)
- Delta model (`motionQA_delta_finetuned_lora_mps`): train loss 0.13, eval loss 0.13 → training SUCCESS.
- Absolute V1 (`motionQaAbsoluteFileLevelRobustScaledClipP95Dec5`): FAILED. Root cause: max-length=1024 → prompt truncated to ~119 tokens.
- Absolute V2 (`motionQaAbsoluteFileLevelV2`): best eval loss 0.6359 (step 1200) → training FAILED (수렴 불가).
  - Loss가 step 100(0.636) 이후 6 epoch 내내 0.60~0.64 정체 → 표현 방식 자체의 학습 한계.
  - 출력 garbage: `[ENDNUM][INT]...` + 한글 문자열 → 포맷 전혀 미학습.
  - Root cause: Robust scaling이 이미 0~1인 절대 좌표에 적용 → [INT]000 항상 고정 → 정수 토큰 무의미 → 학습 신호 부족.
- Do not over-claim from sample-level split only.
- Paper-quality claims must include file-level split results to reduce leakage risk.

## D3. Coordinate Space & Scaling Facts
- MediaPipe Pose Landmarker outputs **normalized coordinates** (x = pixel_x / width, y = pixel_y / height) → base range 0~1, NOT pixel units.
- Both delta and absolute use **robust scaling** (÷ 95th-percentile, clip_c=8.0); ~4.55% of absolute values exceed 1.0 (max ~1.18).
- Inverse scaling is applied automatically at evaluation (default enabled); disable only with `--no-delta-scaling` / `--no-absolute-scaling`.
- Delta scaling: joint별 단일 scale (x/y 공통). Absolute scaling: joint별 x축/y축 별도 scale.

### Scaling 효과 분석 (중요)
- **Delta에 Robust scaling 적합**: delta 값이 0 근처에 몰려 있어 scaling 후 값이 퍼짐 → 정수 토큰 활용 가능 → 학습 신호 증가.
- **Absolute에 Robust scaling 부적합**: MediaPipe 정규화로 이미 0~1 → scaling 후에도 0~1 → [INT]000 항상 고정 → 정수 토큰 무의미.
- Robust scaling 자체는 좋은 기법이나, 이미 0~1인 값에 적용 시 토큰 포맷([INT]ddd) 활용 불가.
- Absolute에 적합한 스케일링 대안: ×100 (0~1 → 0~100, [INT]043 등 활용) 또는 obs[0] 기준 상대 좌표 변환.

## D4. Ablation Study Plan (noScale 실험 - 진행 예정)
### 목적
- "Delta 우수성이 Robust scaling 편향이 아닌 표현 방식 자체의 차이"임을 논문에서 증명하기 위한 ablation.
- 스케일링 없이 두 방법을 동일 조건에서 비교.

### 데이터 생성 (수정 완료)
- `05_delta_convertToFinetuning_fileLevelSplit.py`: `APPLY_ROBUST_SCALING = False` → `finetune_dataset_delta_noScale_fileLevel_*.jsonl`
- `05_abs_convertToFinetuning_fileLevelSplit.py`: `APPLY_ROBUST_SCALING = False` → `finetune_dataset_absolute_noScale_fileLevel_*.jsonl`

### 학습 명령어 (BaseLine/ 폴더)
```
# Delta noScale
python 06_finetune_llama2_lora.py --train-jsonl finetune_dataset_delta_noScale_fileLevel_train.jsonl --val-jsonl finetune_dataset_delta_noScale_fileLevel_val.jsonl --output-dir motionQaDeltaNoScaleFileLevel --max-length 4096 --learning-rate 5e-5 --epochs 6 --batch-size 1 --grad-accum 8 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --token-io-mode lm_head_special_only

# Absolute noScale
python 06_finetune_llama2_lora.py --train-jsonl finetune_dataset_absolute_noScale_fileLevel_train.jsonl --val-jsonl finetune_dataset_absolute_noScale_fileLevel_val.jsonl --output-dir motionQaAbsoluteNoScaleFileLevel --max-length 4096 --learning-rate 5e-5 --epochs 6 --batch-size 1 --grad-accum 8 --lora-r 16 --lora-alpha 32 --lora-dropout 0.05 --token-io-mode lm_head_special_only
```

### 예상 결과 및 논문 서술
| 조건 | Delta loss | Absolute loss | 결론 |
|------|-----------|---------------|------|
| noScale | ? | ? | 표현 방식 자체의 차이 확인 |
| Robust Scaling (현재) | 0.13 ✅ | 0.63 ❌ | Scaling이 Delta에만 유효 |

- noScale에서도 Delta < Absolute이면: "표현 방식 자체가 원인" 주장 강화
- noScale에서 Delta도 수렴 실패면: "Robust scaling이 Delta의 핵심 기여 요소" 주장

### 추론 시 주의 (noScale 모델 사용 시)
- 평가 스크립트에서 `--no-delta-scaling` / `--no-absolute-scaling` 플래그 필수
- scaling_config 없이 raw 값으로 역변환해야 함

## D5. Fair Comparison Protocol (delta vs absolute)
- Both must evaluate in **absolute coordinate space** (MediaPipe normalized, 0~1).
- Delta eval: use both `--append-target-prefix` and `--accumulate-to-absolute` in `delta/01_test2_delta_evalation.py`.
  - Flow: delta_scaled → inverse_scale → accumulate from baseline frame (obs_end_index=10) → MAE.
  - Terminal `mode=abs` is EXPECTED when `--accumulate-to-absolute` is enabled; it means "evaluating in absolute coordinate space", not "using the absolute model."
  - Note: delta has structural accumulation error; absolute does not.
- Absolute eval: `absolute/01_test2_absolute_evalation.py` with `--append-target-prefix`.
  - Flow: coord_scaled → inverse_scale → MAE.
- Both use fileLevel test JSONL for leakage-safe reporting.
- **repetition_penalty 기본값: 1.0** (모든 eval/inference 스크립트). 구조화된 특수 토큰([NUM][INT][SEP][DEC][ENDNUM])이 반복되므로 1.2 이상 설정 시 포맷 붕괴.
- Evaluator/model pairing must stay consistent:
  - delta evaluator + delta LoRA + delta test JSONL
  - absolute evaluator + absolute LoRA + absolute test JSONL
  - never point delta evaluator to `motionQaAbsoluteFileLevelV2` or absolute evaluator to a delta LoRA
- Evaluation commands (Robust Scaling 적용 모델):
  ```
  # absolute (with scaling)
  cd absolute && python 01_test2_absolute_evalation.py --lora-model-dir ../BaseLine/motionQaAbsoluteFileLevelV2 --test-jsonl ../BaseLine/finetune_dataset_absolute_fileLevel_test.jsonl --append-target-prefix --output-report ./absolute_eval_report_v2.json

  # delta (absolute space, with scaling)
  cd delta && python 01_test2_delta_evalation.py --lora-model-dir ../BaseLine/motionQADeltaFileLevelLoraMps --test-jsonl ../BaseLine/finetune_dataset_delta_fileLevel_test.jsonl --append-target-prefix --accumulate-to-absolute --output-report ./delta_eval_report_fileLevel_abs.json
  ```

## D6. Prompt Format Rules (inference 스크립트)
- 학습 포맷: `"Observed motion deltas: ..."` / `"Observed absolute coordinates: ..."` (instruct 스타일 아님)
- inference 시 `build_instruct_prompt_from_observed()` 절대 사용 금지 → 학습 포맷과 불일치 → hallucination
- `delta/01_test2_delta.py`, `absolute/01_test2_absolute.py` 모두 train-format 프롬프트 직접 사용으로 수정 완료
- 생성 시작 전 `\nNext motion deltas:` / `\nNext absolute coordinates:` 접두어 append 필수

## E. Runtime Safety Checklist
- Before training:
  - confirm CUDA (`BaseLine/02_Cuda_available_test.py` or `torch.cuda.is_available()`).
  - confirm model path, tokenizer-extension path, train JSONL path exist.
- Prefer path resolution relative to script location (`__file__`), not shell cwd.
- Keep CLI overrides for model/data/output paths.
- PowerShell multiline command safety:
  - backtick `` ` `` must be the final character on the line; trailing whitespace breaks continuation
  - if continuation breaks, the next `--arg` line is parsed by PowerShell itself and throws `Missing expression after unary operator '--'`
  - safest fallback is to run the command on one line
- After evaluation, inspect saved report `config` block to verify:
  - `lora_model_dir` matches the intended model family
  - `test_jsonl` matches the intended split track
  - delta absolute-space eval has `"accumulate_to_absolute": true`

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
