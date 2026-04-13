import glob
import json
import random
from pathlib import Path

# Input JSON root directory (iterate all action subfolders)
INPUT_ROOT_DIR = "../dataset/new_data"

# Joints used for training
ARM_JOINTS = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
]

# Sliding-window setup (same policy as delta pipeline)
OBS_FRAMES = 10
PRED_FRAMES = 8
STRIDE = 3
WINDOW = OBS_FRAMES + PRED_FRAMES

# Split ratio (train/val/test = 8:1:1)
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

OUTPUT_TRAIN_PATH = "./finetune_dataset_absolute_train.jsonl"
OUTPUT_VAL_PATH = "./finetune_dataset_absolute_val.jsonl"
OUTPUT_TEST_PATH = "./finetune_dataset_absolute_test.jsonl"

# ── Instruct-style prompt ──
INSTRUCT_PROMPT = True

INSTRUCT_SYSTEM = (
    "You are a motion prediction assistant that extrapolates future joint movements "
    "based on observed absolute (x, y) coordinate sequences. "
    "IMPORTANT: Provide EXACTLY 8 frames. Do not provide more or less."
)
INSTRUCT_TEMPLATE = (
    "Forecast the next {pred_len:d} (x, y) absolute coordinates for all observed joints "
    "using the given {obs_len:d} observed coordinate frames.\n"
    "Each coordinate value must follow this token format: -[NUM][INT]000[SEP][DEC]00000[ENDNUM] "
    "(minus sign optional).\n"
    "Return one line in this exact format:\n"
    "Next absolute coordinates: JOINT:(tok,tok),(tok,tok),... | JOINT:(tok,tok),...\n"
    "### Observed Absolute Coordinate Sequences ###\n"
    "{obs_text}"
)

OUTPUT_TRAIN_PATH_INSTRUCT = "./finetune_dataset_absolute_instruct_train.jsonl"
OUTPUT_VAL_PATH_INSTRUCT = "./finetune_dataset_absolute_instruct_val.jsonl"
OUTPUT_TEST_PATH_INSTRUCT = "./finetune_dataset_absolute_instruct_test.jsonl"

# Robust scaling + optional clipping before tokenization (absolute coordinates)
APPLY_ROBUST_SCALING = True
SCALE_PERCENTILE = 95.0
SCALE_EPS = 1e-6
APPLY_CLIPPING = True
CLIP_C = 8.0
SCALING_CONFIG_PATH = "./absolute_scaling_config.json"

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


def build_joint_axis_scales(json_paths: list[str]) -> dict[str, dict[str, float]]:
    vals_x: dict[str, list[float]] = {j: [] for j in ARM_JOINTS}
    vals_y: dict[str, list[float]] = {j: [] for j in ARM_JOINTS}
    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        seq = data.get("pose_sequence", [])
        for frame in seq:
            for joint in ARM_JOINTS:
                if joint in frame:
                    x, y = frame[joint]
                    vals_x[joint].append(abs(x))
                    vals_y[joint].append(abs(y))

    scales: dict[str, dict[str, float]] = {}
    for joint in ARM_JOINTS:
        sx = _percentile(vals_x[joint], SCALE_PERCENTILE)
        sy = _percentile(vals_y[joint], SCALE_PERCENTILE)
        if sx <= SCALE_EPS:
            sx = SCALE_EPS
        if sy <= SCALE_EPS:
            sy = SCALE_EPS
        scales[joint] = {"x": round(sx, 8), "y": round(sy, 8)}
    return scales


def save_scaling_config(path: str, scales: dict[str, dict[str, float]]) -> None:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = (SCRIPT_DIR / out_path).resolve()
    payload = {
        "enabled": APPLY_ROBUST_SCALING,
        "method": "robust_divide_scale_then_clip_absolute",
        "percentile_q": SCALE_PERCENTILE,
        "eps": SCALE_EPS,
        "clip_enabled": APPLY_CLIPPING,
        "clip_c": CLIP_C,
        "joint_axis_scale": scales,
        "note": (
            "coord_scaled = clip(coord_raw / (scale_axis + eps), -c, c); "
            "inverse: coord_raw = coord_scaled * (scale_axis + eps)."
        ),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def scale_coord(
    val: float,
    joint: str,
    axis: str,
    joint_axis_scales: dict[str, dict[str, float]] | None,
) -> float:
    if not joint_axis_scales:
        return round(val, DEC_DIGITS)
    s = joint_axis_scales.get(joint, {}).get(axis, 1.0)
    out = val / (s + SCALE_EPS)
    if APPLY_CLIPPING:
        out = max(-CLIP_C, min(CLIP_C, out))
    return round(out, DEC_DIGITS)


def format_pose_sequence_absolute(
    seq,
    max_frames: int = 10,
    joint_axis_scales: dict[str, dict[str, float]] | None = None,
) -> str:
    if not seq:
        return ""

    formatted = []
    joints_list = [j for j in seq[0].keys() if ARM_JOINTS is None or j in ARM_JOINTS]

    for joint in joints_list:
        coords = []
        for frame in seq[:max_frames]:
            if joint in frame:
                x, y = frame[joint]
                x = scale_coord(x, joint, "x", joint_axis_scales)
                y = scale_coord(y, joint, "y", joint_axis_scales)
                coords.append(f"({num_to_tokens(x)},{num_to_tokens(y)})")

        if coords:
            formatted.append(f"{joint}:{','.join(coords)}")

    return " | ".join(formatted)


def collect_samples(
    json_paths: list[str],
    joint_axis_scales: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    samples = []

    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            seq = data.get("pose_sequence", [])
            if len(seq) < WINDOW + 1:
                continue

            action = Path(path).parent.name
            for start in range(0, len(seq) - WINDOW, STRIDE):
                obs = seq[start : start + OBS_FRAMES]
                pred = seq[start + OBS_FRAMES : start + WINDOW]

                obs_abs = format_pose_sequence_absolute(
                    obs, max_frames=OBS_FRAMES, joint_axis_scales=joint_axis_scales
                )
                pred_abs = format_pose_sequence_absolute(
                    pred, max_frames=PRED_FRAMES, joint_axis_scales=joint_axis_scales
                )
                if not obs_abs or not pred_abs:
                    continue

                if INSTRUCT_PROMPT:
                    body = INSTRUCT_TEMPLATE.format(
                        pred_len=PRED_FRAMES, obs_len=OBS_FRAMES, obs_text=obs_abs
                    )
                    prompt_text = f"{INSTRUCT_SYSTEM}\n\n{body}"
                else:
                    prompt_text = f"Observed absolute coordinates: {obs_abs}"

                samples.append(
                    {
                        "prompt": prompt_text,
                        "completion": f"Next absolute coordinates: {pred_abs}",
                        "task": "trajectory_absolute",
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


def convert_all_to_absolute() -> None:
    json_paths = collect_json_paths()
    if not json_paths:
        raise SystemExit(f"No JSON files found under: {INPUT_ROOT_DIR}")

    joint_axis_scales = None
    if APPLY_ROBUST_SCALING:
        joint_axis_scales = build_joint_axis_scales(json_paths)
        save_scaling_config(SCALING_CONFIG_PATH, joint_axis_scales)
        print(f"[INFO] robust scaling enabled, config saved: {SCALING_CONFIG_PATH}")
        print(f"[INFO] joint-axis scales(q{SCALE_PERCENTILE}): {joint_axis_scales}")
        if APPLY_CLIPPING:
            print(f"[INFO] clipping enabled: c={CLIP_C}")

    samples = collect_samples(json_paths, joint_axis_scales=joint_axis_scales)
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
    convert_all_to_absolute()
