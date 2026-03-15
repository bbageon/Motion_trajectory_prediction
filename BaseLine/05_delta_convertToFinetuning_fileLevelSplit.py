import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# 입력 JSON 루트 폴더 (하위 동작 폴더 전체 순회)
INPUT_ROOT_DIR = "../dataset/new_data"

# 학습에 사용할 관절 (None이면 전체 사용)
ARM_JOINTS = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
]

# Sliding window 설정
OBS_FRAMES = 10
PRED_FRAMES = 8
STRIDE = 3
WINDOW = OBS_FRAMES + PRED_FRAMES

# 파일 단위 분할 비율 (train/val/test = 8:1:1)
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

OUTPUT_TRAIN_PATH = "./finetune_dataset_delta_noScale_fileLevel_train.jsonl"
OUTPUT_VAL_PATH = "./finetune_dataset_delta_noScale_fileLevel_val.jsonl"
OUTPUT_TEST_PATH = "./finetune_dataset_delta_noScale_fileLevel_test.jsonl"

# Robust scaling + optional clipping before tokenization
# 옵션 1: delta' = delta / (s + eps), 중심 이동 없이 0 유지
# 옵션 2: delta' = clip(delta', -c, c)
APPLY_ROBUST_SCALING = False
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
                deltas.append(f"({num_to_tokens(dx)},{num_to_tokens(dy)})")

        if deltas:
            formatted.append(f"{joint}:{','.join(deltas[:max_frames])}")

    return " | ".join(formatted)


def collect_samples_by_file(
    json_paths: list[str],
    joint_scales: dict[str, float] | None = None,
) -> dict[str, list[dict]]:
    samples_by_file: dict[str, list[dict]] = {}

    for path in json_paths:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        seq = data.get("pose_sequence", [])
        if len(seq) < WINDOW + 1:
            continue

        source_file = path.replace("\\", "/")
        action = Path(path).parent.name
        rows: list[dict] = []
        for start in range(0, len(seq) - WINDOW, STRIDE):
            obs = seq[start : start + OBS_FRAMES]
            pred = seq[start + OBS_FRAMES : start + WINDOW]
            obs_deltas = format_pose_sequence_deltas(obs, joint_scales=joint_scales)
            pred_deltas = format_pose_sequence_deltas(pred, joint_scales=joint_scales)
            if not obs_deltas or not pred_deltas:
                continue
            rows.append(
                {
                    "prompt": f"Observed motion deltas: {obs_deltas}",
                    "completion": f"Next motion deltas: {pred_deltas}",
                    "task": "trajectory_delta",
                    "source_file": source_file,
                    "action": action,
                }
            )

        if rows:
            samples_by_file[source_file] = rows

    return samples_by_file


def split_file_paths_stratified(file_paths: list[str], rng: random.Random) -> tuple[set[str], set[str], set[str]]:
    if not file_paths:
        return set(), set(), set()

    paths = file_paths[:]
    rng.shuffle(paths)

    n = len(paths)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    n_test = n - n_train - n_val

    if n >= 3 and n_val == 0:
        n_val = 1
    if n >= 2 and n_test == 0:
        n_test = 1
    if n_train + n_val + n_test > n:
        n_train = max(0, n - n_val - n_test)

    train = set(paths[:n_train])
    val = set(paths[n_train : n_train + n_val])
    test = set(paths[n_train + n_val : n_train + n_val + n_test])
    return train, val, test


def split_by_file(samples_by_file: dict[str, list[dict]]) -> tuple[list[dict], list[dict], list[dict], dict[str, set[str]]]:
    # action별 파일 리스트를 만든 뒤, action 단위로 파일 분할(약한 stratified split)
    files_by_action: dict[str, list[str]] = defaultdict(list)
    for source_file, rows in samples_by_file.items():
        action = rows[0]["action"]
        files_by_action[action].append(source_file)

    rng = random.Random(RANDOM_SEED)
    split_files = {"train": set(), "val": set(), "test": set()}
    for action, file_paths in files_by_action.items():
        train_f, val_f, test_f = split_file_paths_stratified(file_paths, rng)
        split_files["train"].update(train_f)
        split_files["val"].update(val_f)
        split_files["test"].update(test_f)

    # 안전 체크: 파일 겹침 없어야 함
    if split_files["train"] & split_files["val"]:
        raise RuntimeError("File overlap detected between train and val.")
    if split_files["train"] & split_files["test"]:
        raise RuntimeError("File overlap detected between train and test.")
    if split_files["val"] & split_files["test"]:
        raise RuntimeError("File overlap detected between val and test.")

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    test_rows: list[dict] = []
    for source_file, rows in samples_by_file.items():
        if source_file in split_files["train"]:
            train_rows.extend(rows)
        elif source_file in split_files["val"]:
            val_rows.extend(rows)
        elif source_file in split_files["test"]:
            test_rows.extend(rows)

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    rng.shuffle(test_rows)
    return train_rows, val_rows, test_rows, split_files


def write_jsonl(path: str, rows: list[dict]) -> None:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = (SCRIPT_DIR / out_path).resolve()
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def action_counter(rows: list[dict]) -> dict[str, int]:
    c = Counter(row.get("action", "unknown") for row in rows)
    return dict(sorted(c.items()))


def convert_all_to_delta_file_level() -> None:
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

    samples_by_file = collect_samples_by_file(json_paths, joint_scales=joint_scales)
    train_rows, val_rows, test_rows, split_files = split_by_file(samples_by_file)

    write_jsonl(OUTPUT_TRAIN_PATH, train_rows)
    write_jsonl(OUTPUT_VAL_PATH, val_rows)
    write_jsonl(OUTPUT_TEST_PATH, test_rows)

    total_samples = len(train_rows) + len(val_rows) + len(test_rows)
    print(
        f"[DONE:file-level] total_samples={total_samples} | "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )
    print(
        f"[FILES] train={len(split_files['train'])} "
        f"val={len(split_files['val'])} test={len(split_files['test'])}"
    )
    print(f"[ACTIONS] train={action_counter(train_rows)}")
    print(f"[ACTIONS] val  ={action_counter(val_rows)}")
    print(f"[ACTIONS] test ={action_counter(test_rows)}")
    print(f" - train: {OUTPUT_TRAIN_PATH}")
    print(f" - val  : {OUTPUT_VAL_PATH}")
    print(f" - test : {OUTPUT_TEST_PATH}")


if __name__ == "__main__":
    convert_all_to_delta_file_level()
