import glob
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Input JSON root directory
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

# Sliding-window setup
OBS_FRAMES = 10
PRED_FRAMES = 8
STRIDE = 3
WINDOW = OBS_FRAMES + PRED_FRAMES

# File-level split ratio (train/val/test = 8:1:1)
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

OUTPUT_TRAIN_PATH = "./x100 데이터/finetune_dataset_absolute_x100_fileLevel_train.jsonl"
OUTPUT_VAL_PATH = "./x100 데이터/finetune_dataset_absolute_x100_fileLevel_val.jsonl"
OUTPUT_TEST_PATH = "./x100 데이터/finetune_dataset_absolute_x100_fileLevel_test.jsonl"

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

OUTPUT_TRAIN_PATH_INSTRUCT = "./x100 데이터/finetune_dataset_absolute_x100_instruct_fileLevel_train.jsonl"
OUTPUT_VAL_PATH_INSTRUCT = "./x100 데이터/finetune_dataset_absolute_x100_instruct_fileLevel_val.jsonl"
OUTPUT_TEST_PATH_INSTRUCT = "./x100 데이터/finetune_dataset_absolute_x100_instruct_fileLevel_test.jsonl"

# ×100 선형 스케일: MediaPipe 정규화 좌표(0~1)에 SCALE_FACTOR를 곱해
# [INT] 토큰이 000~100 범위로 다양해짐 → delta와 공평한 비교 조건
APPLY_ROBUST_SCALING = False
SCALE_FACTOR = 100.0  # coord × 100 → [INT]가 000~102 범위로 다양해짐

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


def collect_json_paths() -> list[str]:
    input_root = Path(INPUT_ROOT_DIR)
    if not input_root.is_absolute():
        input_root = (SCRIPT_DIR / input_root).resolve()
    pattern = str(input_root / "**" / "*.json")
    return sorted(glob.glob(pattern, recursive=True))


def apply_x100(val: float) -> float:
    """좌표값에 ×100 적용 후 반올림."""
    return round(val * SCALE_FACTOR, DEC_DIGITS)


def format_pose_sequence_absolute(seq, max_frames: int = 10) -> str:
    if not seq:
        return ""

    formatted = []
    joints_list = [j for j in seq[0].keys() if ARM_JOINTS is None or j in ARM_JOINTS]

    for joint in joints_list:
        coords = []
        for frame in seq[:max_frames]:
            if joint in frame:
                x, y = frame[joint]
                x = apply_x100(x)
                y = apply_x100(y)
                coords.append(f"({num_to_tokens(x)},{num_to_tokens(y)})")

        if coords:
            formatted.append(f"{joint}:{','.join(coords)}")

    return " | ".join(formatted)


def collect_samples_by_file(json_paths: list[str]) -> dict[str, list[dict]]:
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
            obs_abs = format_pose_sequence_absolute(obs, max_frames=OBS_FRAMES)
            pred_abs = format_pose_sequence_absolute(pred, max_frames=PRED_FRAMES)
            if not obs_abs or not pred_abs:
                continue

            if INSTRUCT_PROMPT:
                body = INSTRUCT_TEMPLATE.format(
                    pred_len=PRED_FRAMES, obs_len=OBS_FRAMES, obs_text=obs_abs
                )
                prompt_text = f"{INSTRUCT_SYSTEM}\n\n{body}"
            else:
                prompt_text = f"Observed absolute coordinates: {obs_abs}"

            rows.append(
                {
                    "prompt": prompt_text,
                    "completion": f"Next absolute coordinates: {pred_abs}",
                    "task": "trajectory_absolute",
                    "source_file": source_file,
                    "action": action,
                }
            )

        if rows:
            samples_by_file[source_file] = rows

    return samples_by_file


def split_file_paths_stratified(
    file_paths: list[str], rng: random.Random
) -> tuple[set[str], set[str], set[str]]:
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


def split_by_file(
    samples_by_file: dict[str, list[dict]],
) -> tuple[list[dict], list[dict], list[dict], dict[str, set[str]]]:
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def action_counter(rows: list[dict]) -> dict[str, int]:
    c = Counter(row.get("action", "unknown") for row in rows)
    return dict(sorted(c.items()))


def convert_all_to_absolute_x100_file_level() -> None:
    json_paths = collect_json_paths()
    if not json_paths:
        raise SystemExit(f"No JSON files found under: {INPUT_ROOT_DIR}")

    print(f"[INFO] x100 linear scaling: coord × {SCALE_FACTOR} (no Robust Scaling)")
    print(f"[INFO] [INT] token range after x100: 000~102 (MediaPipe coords 0~1.02)")

    samples_by_file = collect_samples_by_file(json_paths)
    train_rows, val_rows, test_rows, split_files = split_by_file(samples_by_file)

    t_path = OUTPUT_TRAIN_PATH_INSTRUCT if INSTRUCT_PROMPT else OUTPUT_TRAIN_PATH
    v_path = OUTPUT_VAL_PATH_INSTRUCT if INSTRUCT_PROMPT else OUTPUT_VAL_PATH
    e_path = OUTPUT_TEST_PATH_INSTRUCT if INSTRUCT_PROMPT else OUTPUT_TEST_PATH

    write_jsonl(t_path, train_rows)
    write_jsonl(v_path, val_rows)
    write_jsonl(e_path, test_rows)

    total_samples = len(train_rows) + len(val_rows) + len(test_rows)
    prompt_mode = "instruct" if INSTRUCT_PROMPT else "plain"
    print(
        f"[DONE:file-level] total_samples={total_samples} | "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)} | "
        f"prompt_mode={prompt_mode}"
    )
    print(
        f"[FILES] train={len(split_files['train'])} "
        f"val={len(split_files['val'])} test={len(split_files['test'])}"
    )
    print(f"[ACTIONS] train={action_counter(train_rows)}")
    print(f"[ACTIONS] val  ={action_counter(val_rows)}")
    print(f"[ACTIONS] test ={action_counter(test_rows)}")
    print(f" - train: {t_path}")
    print(f" - val  : {v_path}")
    print(f" - test : {e_path}")


if __name__ == "__main__":
    convert_all_to_absolute_x100_file_level()
