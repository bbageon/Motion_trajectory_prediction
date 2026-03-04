import glob
import json
import os
import random
from pathlib import Path

# 입력 JSON 루트 폴더 (하위 동작 폴더 전체 순회)
INPUT_ROOT_DIR = "../dataset"
# 분할 비율 (train/val/test = 8:1:1)
SPLIT_RATIOS = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

OUTPUT_TRAIN_PATH = "./finetune_dataset_delta_train.jsonl"
OUTPUT_VAL_PATH = "./finetune_dataset_delta_val.jsonl"
OUTPUT_TEST_PATH = "./finetune_dataset_delta_test.jsonl"

SCRIPT_DIR = Path(__file__).resolve().parent


def num_to_tokens(val: float) -> str:
    sign = "-" if val < 0 else ""
    val = abs(val)
    int_part = int(val)
    dec_part = round((val - int_part) * 1000)
    return (
        f"{sign}[NUM][INT]{str(int_part).zfill(3)}"
        f"[SEP][DEC]{str(dec_part).zfill(3)}[ENDNUM]"
    )


def format_pose_sequence_deltas(seq, max_frames: int = 10) -> str:
    if len(seq) < 2:
        return ""

    formatted = []
    joints_list = seq[0].keys()

    for joint in joints_list:
        deltas = []
        for i in range(1, len(seq)):
            if joint in seq[i] and joint in seq[i - 1]:
                x1, y1 = seq[i - 1][joint]
                x2, y2 = seq[i][joint]
                dx = round(x2 - x1, 3)
                dy = round(y2 - y1, 3)
                dx_tok = num_to_tokens(dx)
                dy_tok = num_to_tokens(dy)
                deltas.append(f"({dx_tok},{dy_tok})")

        if deltas:
            formatted.append(f"{joint}:{','.join(deltas[:max_frames])}")

    return " | ".join(formatted)


def collect_samples() -> list[dict]:
    samples = []
    input_root = Path(INPUT_ROOT_DIR)
    if not input_root.is_absolute():
        input_root = (SCRIPT_DIR / input_root).resolve()
    pattern = str(input_root / "**" / "*.json")

    for path in glob.glob(pattern, recursive=True):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
            seq = data.get("pose_sequence", [])
            if len(seq) < 3:
                continue

            mid = len(seq) // 2
            obs = seq[:mid]
            pred = seq[mid:]

            obs_deltas = format_pose_sequence_deltas(obs)
            pred_deltas = format_pose_sequence_deltas(pred)
            if not obs_deltas or not pred_deltas:
                continue

            action = Path(path).parent.name
            samples.append(
                {
                    "prompt": f"Observed motion deltas: {obs_deltas}",
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
    samples = collect_samples()
    train_rows, val_rows, test_rows = split_samples(samples)

    write_jsonl(OUTPUT_TRAIN_PATH, train_rows)
    write_jsonl(OUTPUT_VAL_PATH, val_rows)
    write_jsonl(OUTPUT_TEST_PATH, test_rows)

    print(
        f"[DONE] total={len(samples)} | "
        f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}"
    )
    print(f" - train: {OUTPUT_TRAIN_PATH}")
    print(f" - val  : {OUTPUT_VAL_PATH}")
    print(f" - test : {OUTPUT_TEST_PATH}")


if __name__ == "__main__":
    convert_all_to_delta()
