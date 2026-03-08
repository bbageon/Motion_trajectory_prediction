"""
absolute/02_eval_with_dataset.py
MPS Delta LoRA 모델로 delta를 예측한 뒤 절대 좌표로 누적하여 MAE를 계산합니다.

주의: CPU에서 실행 시 샘플당 수십 분이 소요됩니다.
      GPU(CUDA) 또는 MPS 환경에서 실행을 권장합니다.
"""
import json
import re
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ----------------------------
# 경로 설정 (absolute/ 폴더에서 실행 기준)
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(BASE_DIR, "..")

base_model_dir  = os.path.join(ROOT_DIR, "Meta-Llama-3.1-8B_tokenizerExtension")
lora_model_dir  = os.path.join(ROOT_DIR, "BaseLine", "motionQA_delta_finetuned_lora_mps")
test_jsonl_path = os.path.join(ROOT_DIR, "BaseLine", "finetune_dataset_delta_test.jsonl")

# ----------------------------
# 1 모델 & 토크나이저 로드
# ----------------------------
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(base_model_dir, local_files_only=True)
print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, lora_model_dir)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).to(torch.float32)
print(f"Model ready on {device}")

# ----------------------------
# 2 토큰 파싱 (JSONL 포맷: [NUM][INT]000[SEP][DEC]00000[ENDNUM])
# ----------------------------
TOKEN_PATTERN = re.compile(
    r"(-?)\[NUM\]\[INT\](\d{3})\[SEP\]\[DEC\](\d{5})\[ENDNUM\]"
)
DEC_SCALE = 100000

def decode_token(s):
    m = TOKEN_PATTERN.search(s)
    if not m:
        raise ValueError(f"Invalid token: {s!r}")
    sign = -1 if m.group(1) == "-" else 1
    return round(sign * (int(m.group(2)) + int(m.group(3)) / DEC_SCALE), 6)


def parse_delta_sequence(text):
    pat = re.compile(
        r"([A-Z_]+):\((-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{5}\[ENDNUM\])"
        r",(-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{5}\[ENDNUM\])\)"
    )
    result = {}
    for joint, x_tok, y_tok in pat.findall(text):
        result.setdefault(joint, []).append((decode_token(x_tok), decode_token(y_tok)))
    return result


def delta_to_absolute(baseline_frame, deltas):
    """delta 시퀀스를 절대 좌표로 누적 변환"""
    abs_seq = {}
    for joint, seq in deltas.items():
        if joint not in baseline_frame:
            continue
        bx, by = baseline_frame[joint]
        x, y = bx, by
        coords = []
        for dx, dy in seq:
            x = round(x + dx, 4)
            y = round(y + dy, 4)
            coords.append((x, y))
        abs_seq[joint] = coords
    return abs_seq


# ----------------------------
# 3 MAE
# ----------------------------
def compute_mae(pred, gt):
    errors = []
    for joint in gt:
        if joint not in pred:
            continue
        n = min(len(gt[joint]), len(pred[joint]))
        for i in range(n):
            errors.append(abs(pred[joint][i][0] - gt[joint][i][0]))
            errors.append(abs(pred[joint][i][1] - gt[joint][i][1]))
    return sum(errors) / len(errors) if errors else float("nan")


# ----------------------------
# 4 테스트 실행
# ----------------------------
print(f"\nTest data: {test_jsonl_path}")
with open(test_jsonl_path, encoding="utf-8") as f:
    test_samples = [json.loads(line) for line in f]
print(f"Samples: {len(test_samples)}\n{'='*60}")

mae_list = []

for idx, sample in enumerate(test_samples):
    prompt   = sample["prompt"]
    gt_text  = sample["completion"]
    src_file = sample.get("source_file", "")
    action   = sample.get("action", "?")
    fname    = os.path.basename(src_file)

    print(f"\n[{idx+1}/{len(test_samples)}] {action} | {fname}")

    # 원본 JSON에서 마지막 관측 프레임(10번째) 절대 좌표 로드
    baseline_frame = {}
    try:
        with open(src_file, encoding="utf-8") as jf:
            raw = json.load(jf)
        pose_seq = raw["pose_sequence"]
        obs_end  = min(10, len(pose_seq) - 1)
        baseline_frame = {k: tuple(v) for k, v in pose_seq[obs_end].items()}
        print(f"  Baseline loaded from frame {obs_end} ({len(baseline_frame)} joints)")
    except Exception as e:
        print(f"  WARNING: Could not load source JSON: {e}")

    inputs = tokenizer(
        prompt, return_tensors="pt",
        truncation=True, max_length=6144
    ).to(device)
    input_len = inputs["input_ids"].shape[1]
    print(f"  Input tokens: {input_len}")

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=6144,
            do_sample=True,
            temperature=0.2,
            top_p=0.9,
        )

    new_ids   = out[0][input_len:]
    pred_text = tokenizer.decode(new_ids, skip_special_tokens=False)
    print(f"  Pred tokens generated: {len(new_ids)}")

    try:
        pred_deltas = parse_delta_sequence(pred_text)
        gt_deltas   = parse_delta_sequence(gt_text)

        if not pred_deltas:
            print("  WARNING: No parseable tokens in prediction.")
            continue
        if not baseline_frame:
            print("  WARNING: No baseline; skipping absolute MAE.")
            continue

        # delta -> absolute 누적
        pred_abs = delta_to_absolute(baseline_frame, pred_deltas)
        gt_abs   = delta_to_absolute(baseline_frame, gt_deltas)

        mae = compute_mae(pred_abs, gt_abs)
        mae_list.append(mae)

        p_frames = max(len(v) for v in pred_abs.values()) if pred_abs else 0
        g_frames = max(len(v) for v in gt_abs.values())   if gt_abs   else 0
        print(f"  Pred: {len(pred_abs)} joints, {p_frames} frames")
        print(f"  GT:   {len(gt_abs)} joints, {g_frames} frames")
        print(f"  Absolute MAE: {mae:.6f}")

    except Exception as e:
        print(f"  ERROR: {e}")

print(f"\n{'='*60}")
print("[Evaluation Summary - Absolute (via Delta Accumulation)]")
print(f"  Evaluated: {len(mae_list)}/{len(test_samples)} samples")
if mae_list:
    print(f"  Mean Abs MAE : {sum(mae_list)/len(mae_list):.6f}")
    print(f"  Min MAE      : {min(mae_list):.6f}")
    print(f"  Max MAE      : {max(mae_list):.6f}")
print("="*60)
