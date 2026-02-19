# Motion QA Pipeline (MP4 -> Pose JSON -> Finetuning JSONL)

This repository builds motion datasets from videos and prepares finetuning data for:
- Delta trajectory modeling
- Absolute trajectory modeling

It also includes inference/visualization scripts for both modes.

## 1. Environment Setup

Use Python virtual environment first.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

This project is stabilized on `mediapipe==0.10.32` (tasks API).
If installation fails, use Python 3.10 or 3.11.

Reset command:

```powershell
pip uninstall -y mediapipe
pip install mediapipe==0.10.32
```

## 2. File Roles

- `mp4ToJson.py`
  - Reads `*.mp4`
  - Extracts MediaPipe pose landmarks
  - Saves compatible `pose_sequence` JSON files
  - Includes JSON schema/compatibility validation

- `04_delta_convertToFinetuning2.py`
  - Converts `pose_sequence` JSON to delta finetuning dataset
  - Output: `finetune_dataset_delta.jsonl`

- `04_abs_covertToFinetuning2.py`
  - Converts `pose_sequence` JSON to absolute finetuning dataset
  - Output: `finetune_dataset_absolute.jsonl`

- `03_TokenizerExtend.py`
  - Extends tokenizer/model vocab for custom motion tokens

- `01_offlineDownload.py`
  - Downloads base model snapshot from Hugging Face

- `02_Cuda_available_test.py`
  - Quick CUDA availability check

- `delta/01_test2_delta.py`
  - Delta-mode inference + parsing + visualization + JSON/GIF save

- `absolute/01_test2_absolute.py`
  - Absolute-mode inference + parsing + visualization + JSON/GIF save

## 3. Recommended Execution Order

### Step 1) (Optional) Download base model

```powershell
python 01_offlineDownload.py
```

### Step 2) (Optional) Check CUDA

```powershell
python 02_Cuda_available_test.py
```

### Step 3) (Optional) Extend tokenizer/model for custom tokens

```powershell
python 03_TokenizerExtend.py
```

### Step 4) Convert MP4 to pose JSON

Input MP4 default folder is `./Custom_Data`.

```powershell
python mp4ToJson.py --input-dir ./Custom_Data --output-dir ./Custom_Data
```

`mp4ToJson.py` will auto-download `pose_landmarker_lite.task` to `./models/` if missing.
If auto-download is blocked, pass `.task` path manually:

```powershell
python mp4ToJson.py --input-dir ./Custom_Data --output-dir ./Custom_Data --task-model ./pose_landmarker_lite.task
```

Strict JSON-only validation:

```powershell
python mp4ToJson.py --validate-json-only --output-dir ./Custom_Data
```

### Step 5) Build finetuning JSONL datasets

Delta dataset:

```powershell
python 04_delta_convertToFinetuning2.py
```

Absolute dataset:

```powershell
python 04_abs_covertToFinetuning2.py
```

### Step 6) Finetune LoRA adapter

Delta finetuning:

```powershell
python 06_finetune_llama2_lora.py --train-jsonl ./finetune_dataset_delta.jsonl --output-dir ./motionQA_delta_finetuned_lora_mps
```

Absolute finetuning:

```powershell
python 06_finetune_llama2_lora.py --train-jsonl ./finetune_dataset_absolute.jsonl --output-dir ./motionQA_finetuned_lora_mps
```

### Step 7) Run inference tests (after finetuning models are ready)

Delta test:

```powershell
python delta\01_test2_delta.py
```

Absolute test:

```powershell
python absolute\01_test2_absolute.py
```

## 4. Data Format (Compatibility Target)

All conversion and test scripts expect this JSON format:

```json
{
  "pose_sequence": [
    {
      "RIGHT_SHOULDER": [0.123, 0.456],
      "RIGHT_ELBOW": [0.130, 0.500],
      "RIGHT_WRIST": [0.140, 0.560],
      "LEFT_SHOULDER": [0.220, 0.460],
      "LEFT_ELBOW": [0.230, 0.510],
      "LEFT_WRIST": [0.240, 0.570]
    }
  ]
}
```

`mp4ToJson.py` validates this structure before saving.

## 5. Notes

- Current finetuning conversion scripts read `./Custom_Data/*.json`.
- Test scripts use hardcoded JSON/model paths; update those paths before running.
- If you use GPU, install a CUDA-compatible `torch` build matching your environment.
