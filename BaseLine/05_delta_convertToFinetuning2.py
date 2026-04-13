import glob
import json
import random
from pathlib import Path

# 입력 JSON 루트 폴더 (하위 동작 폴더 전체 순회)
INPUT_ROOT_DIR = "../dataset/new_data"

# 학습에 사용할 관절 (None이면 전체 사용)
ARM_JOINTS = [
    "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
    "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
]
# Sliding window 설정
OBS_FRAMES = 10   # 관측 프레임 수
PRED_FRAMES = 8   # 예측 프레임 수
STRIDE = 3        # 윈도우 이동 간격 (작을수록 샘플 많아짐)
WINDOW = OBS_FRAMES + PRED_FRAMES  # 총 윈도우 크기

# 분할 비율 (train/val/test = 8:1:1)
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

OUTPUT_TRAIN_PATH = "./finetune_dataset_delta_train.jsonl"
OUTPUT_VAL_PATH = "./finetune_dataset_delta_val.jsonl"
OUTPUT_TEST_PATH = "./finetune_dataset_delta_test.jsonl"

# ── Instruct-style prompt ──
INSTRUCT_PROMPT = True

INSTRUCT_SYSTEM = (
    "You are a motion prediction assistant that extrapolates future joint movements "
    "based on observed delta (dx, dy) coordinate sequences. "
    "IMPORTANT: Provide EXACTLY 8 frames. Do not provide more or less."
)
INSTRUCT_TEMPLATE = (
    "Forecast the next {pred_len:d} (x, y) coordinate deltas for all observed joints "
    "using the given {obs_len:d} observed delta frames.\n"
    "Each delta value must follow this token format: -[NUM][INT]000[SEP][DEC]00000[ENDNUM] "
    "(minus sign optional).\n"
    "Return one line in this exact format:\n"
    "Next motion deltas: JOINT:(tok,tok),(tok,tok),... | JOINT:(tok,tok),...\n"
    "### Observed Delta Sequences ###\n"
    "{obs_text}"
)

OUTPUT_TRAIN_PATH_INSTRUCT = "./finetune_dataset_delta_instruct_train.jsonl"
OUTPUT_VAL_PATH_INSTRUCT = "./finetune_dataset_delta_instruct_val.jsonl"
OUTPUT_TEST_PATH_INSTRUCT = "./finetune_dataset_delta_instruct_test.jsonl"

# Robust scaling + optional clipping before tokenization
# 옵션 1: delta' = delta / (s + eps), 중심 이동 없이 0 유지
# 옵션 2: delta' = clip(delta', -c, c)
APPLY_ROBUST_SCALING = True
SCALE_PERCENTILE = 95.0
SCALE_EPS = 1e-6
APPLY_CLIPPING = True
CLIP_C = 8.0
SCALING_CONFIG_PATH = "./delta_scaling_config.json"

SCRIPT_DIR = Path(__file__).resolve().parent
DEC_DIGITS = 5
DEC_SCALE = 10**DEC_DIGITS


def num_to_tokens(val: float) -> str:
    sign = "-" if val < 0 else ""
    val = abs(val)
    int_part = int(val)
    dec_part = int(round((val - int_part) * DEC_SCALE))
    if dec_part >= DEC_SCALE:
        int_part += 1
        dec_part -= DEC_SCALE
    return (
        f"{sign}[NUM][INT]{str(int_part).zfill(3)}"
        f"[SEP][DEC]{str(dec_part).zfill(DEC_DIGITS)}[ENDNUM]"
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 100:
        return max(values)
    arr = sorted(values)
    pos = (len(arr) - 1) * (q / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(arr) - 1)
    if lo == hi:
        return arr[lo]
    frac = pos - lo
    return arr[lo] * (1.0 - frac) + arr[hi] * frac


def collect_json_paths() -> list[str]:
    input_root = Path(INPUT_ROOT_DIR)
    if not input_root.is_absolute():
        input_root = (SCRIPT_DIR / input_root).resolve()
    pattern = str(input_root / "**" / "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def build_joint_scales(json_paths: list[str]) -> dict[str, float]:
    abs_deltas: dict[str, list[float]] = {j: [] for j in ARM_JOINTS}
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        seq = data.get("pose_sequence", [])
        if len(seq) < 2:
            continue
        for i in range(1, len(seq)):
            prev_f = seq[i - 1]
            cur_f = seq[i]
            for joint in ARM_JOINTS:
                if joint in prev_f and joint in cur_f:
                    x1, y1 = prev_f[joint]
                    x2, y2 = cur_f[joint]
                    abs_deltas[joint].append(abs(x2 - x1))
                    abs_deltas[joint].append(abs(y2 - y1))

    scales: dict[str, float] = {}
    for joint in ARM_JOINTS:
        s = _percentile(abs_deltas[joint], SCALE_PERCENTILE)
        if s <= SCALE_EPS:
            s = SCALE_EPS
        scales[joint] = round(s, 8)
    return scales


def save_scaling_config(path: str, scales: dict[str, float]) -> None:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = (SCRIPT_DIR / out_path).resolve()
    gains_compat = {j: (1.0 / (s + SCALE_EPS)) for j, s in scales.items()}
    payload = {
        "enabled": APPLY_ROBUST_SCALING,
        "method": "robust_divide_scale_then_clip",
        "percentile_q": SCALE_PERCENTILE,
        "eps": SCALE_EPS,
        "clip_enabled": APPLY_CLIPPING,
        "clip_c": CLIP_C,
        "joint_scale": scales,
        # backward compatibility for scripts that still read joint_gain.
        "joint_gain": gains_compat,
        "note": (
            "delta_scaled = clip(delta_raw / (scale + eps), -c, c); "
            "inverse: delta_raw = delta_scaled * (scale + eps)."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def scale_delta(val: float, joint: str, joint_scales: dict[str, float] | None) -> float:
    if not joint_scales:
        return round(val, DEC_DIGITS)
    s = joint_scales.get(joint, 1.0)
    scaled = val / (s + SCALE_EPS)
    if APPLY_CLIPPING:
        scaled = max(-CLIP_C, min(CLIP_C, scaled))
    return round(scaled, DEC_DIGITS)


def format_pose_sequence_deltas(
    seq,
    max_frames: int = 10,
    joint_scales: dict[str, float] | None = None,
) -> str:
    if len(seq) < 2:
        return ""

    formatted = []
    joints_list = [j for j in seq[0].keys() if ARM_JOINTS is None or j in ARM_JOINTS]

    for joint in joints_list:
        deltas = []
        for i in range(1, len(seq)):
            if joint in seq[i] and joint in seq[i - 1]:
                x1, y1 = seq[i - 1][joint]
                x2, y2 = seq[i][joint]
                dx_raw = x2 - x1
                dy_raw = y2 - y1
                dx = scale_delta(dx_raw, joint, joint_scales)
                dy = scale_delta(dy_raw, joint, joint_scales)
                dx_tok = num_to_tokens(dx)
                dy_tok = num_to_tokens(dy)
                deltas.append(f"({dx_tok},{dy_tok})")

        if deltas:
            formatted.append(f"{joint}:{','.join(deltas[:max_frames])}")

    return " | ".join(formatted)


def collect_samples(json_paths: list[str], joint_scales: dict[str, float] | None = None) -> list[dict]:
    samples = []

    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            seq = data.get("pose_sequence", [])
            if len(seq) < WINDOW + 1:
                continue

            action = Path(path).parent.name
            # sliding window: stride 간격으로 윈도우를 이동하며 샘플 추출
            for start in range(0, len(seq) - WINDOW, STRIDE):
                obs = seq[start : start + OBS_FRAMES]
                pred = seq[start + OBS_FRAMES : start + WINDOW]

                obs_deltas = format_pose_sequence_deltas(obs, joint_scales=joint_scales)
                pred_deltas = format_pose_sequence_deltas(pred, joint_scales=joint_scales)
                if not obs_deltas or not pred_deltas:
                    continue

                if INSTRUCT_PROMPT:
                    body = INSTRUCT_TEMPLATE.format(
                        pred_len=PRED_FRAMES, obs_len=OBS_FRAMES, obs_text=obs_deltas
                    )
                    prompt_text = f"{INSTRUCT_SYSTEM}\n\n{body}"
                else:
                    prompt_text = f"Observed motion deltas: {obs_deltas}"

                samples.append(
                    {
                        "prompt": prompt_text,
                        "completion": f"Next motion deltas: {pred_deltas}",
                        "task": "trajectory_delta",
                        "source_file": path.replace("\\", "/"),
                        "action": action,
                    }
                )

    return samples


def split_samples(samples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if not samples:
        return [], [], []

    random.seed(RANDOM_SEED)
    random.shuffle(samples)

    n = len(samples)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    n_test = n - n_train - n_val

    if n >= 3 and n_val == 0:
        n_val = 1
    if n >= 2 and n_test == 0:
        n_test = 1
    if n_train + n_val + n_test > n:
        n_train = max(0, n - n_val - n_test)

    train = samples[:n_train]
    val = samples[n_train : n_train + n_val]
    test = samples[n_train + n_val : n_train + n_val + n_test]
    return train, val, test


def write_jsonl(path: str, rows: list[dict]) -> None:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = (SCRIPT_DIR / out_path).resolve()
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_all_to_delta() -> None:
    json_paths = collect_json_paths()
    if not json_paths:
        raise SystemExit(f"No JSON files found under: {INPUT_ROOT_DIR}")

    joint_scales = None
    if APPLY_ROBUST_SCALING:
        joint_scales = build_joint_scales(json_paths)
        save_scaling_config(SCALING_CONFIG_PATH, joint_scales)
        print(f"[INFO] robust scaling enabled, config saved: {SCALING_CONFIG_PATH}")
        print(f"[INFO] joint scales(q{SCALE_PERCENTILE}): {joint_scales}")
        if APPLY_CLIPPING:
            print(f"[INFO] clipping enabled: c={CLIP_C}")

    samples = collect_samples(json_paths, joint_scales=joint_scales)
    train_rows, val_rows, test_rows = split_samples(samples)

    t_path = OUTPUT_TRAIN_PATH_INSTRUCT if INSTRUCT_PROMPT else OUTPUT_TRAIN_PATH
    v_path = OUTPUT_VAL_PATH_INSTRUCT if INSTRUCT_PROMPT else OUTPUT_VAL_PATH
    e_path = OUTPUT_TEST_PATH_INSTRUCT if INSTRUCT_PROMPT else OUTPUT_TEST_PATH

    write_jsonl(t_path, train_rows)
    write_jsonl(v_path, val_rows)
    write_jsonl(e_path, test_rows)

    prompt_mode = "instruct" if INSTRUCT_PROMPT else "plain"
    print(
        f"[DONE] total={len(samples)} | "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} | "
        f"prompt_mode={prompt_mode}"
    )
    print(f" - train: {t_path}")
    print(f" - val  : {v_path}")
    print(f" - test : {e_path}")


if __name__ == "__main__":
    convert_all_to_delta()
