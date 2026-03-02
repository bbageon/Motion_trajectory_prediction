import glob
import json
import os

# 입력 JSON 파일 폴더
INPUT_DIR = "./Custom_Data"
# 변환 결과(JSONL) 저장 경로
OUTPUT_PATH = "./finetune_dataset_absolute.jsonl"

# absolute 파이프라인에서 사용하는 상체 관절
ARM_JOINTS = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
]


def num_to_tokens(val: float) -> str:
    """
    Convert a float to the structured numeric token format.
    Example: -0.012 -> [NUM]-[DEC]000[SEP][DEC]012[ENDNUM]
    """
    sign = "-" if val < 0 else ""
    val = abs(val)
    int_part = int(val)
    dec_part = int(round((val - int_part) * 1000))
    return f"[NUM]{sign}[DEC]{int_part:03d}[SEP][DEC]{dec_part:03d}[ENDNUM]"


def normalize_absolute_sequence(seq):
    """
    Keep only frames where all ARM_JOINTS are present.
    """
    absolute_seq = []
    for frame in seq:
        frame_coords = {}
        valid_frame = True
        for joint in ARM_JOINTS:
            if joint not in frame:
                valid_frame = False
                break
            x, y = frame[joint]
            frame_coords[joint] = (round(x, 4), round(y, 4))
        if valid_frame:
            absolute_seq.append(frame_coords)
    return absolute_seq


def format_pose_sequence_absolute(seq, max_frames: int = 10) -> str:
    """
    Encode absolute pose sequence as frame-wise joint:(x_token,y_token) text.
    """
    if not seq:
        return ""

    lines = []
    for i, frame in enumerate(seq[:max_frames]):
        parts = []
        for joint in ARM_JOINTS:
            x, y = frame[joint]
            x_tok = num_to_tokens(x)
            y_tok = num_to_tokens(y)
            parts.append(f"{joint}:({x_tok},{y_tok})")
        lines.append(f"Frame[{i}]: " + " | ".join(parts))

    return "\n".join(lines)


def convert_all_to_absolute() -> None:
    """
    Read [INPUT_DIR]/*.json and build JSONL finetuning samples for absolute coordinates.
    """
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fout:
        for path in glob.glob(os.path.join(INPUT_DIR, "*.json")):
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                seq = data.get("pose_sequence", [])
                abs_seq = normalize_absolute_sequence(seq)

                if len(abs_seq) < 2:
                    continue

                mid = len(abs_seq) // 2
                obs = abs_seq[:mid]
                pred = abs_seq[mid:]

                obs_abs = format_pose_sequence_absolute(obs)
                pred_abs = format_pose_sequence_absolute(pred)
                if not obs_abs or not pred_abs:
                    continue

                prompt = f"Observed absolute coordinates:\n{obs_abs}"
                completion = f"Next absolute coordinates:\n{pred_abs}"

                fout.write(
                    json.dumps(
                        {
                            "prompt": prompt,
                            "completion": completion,
                            "task": "trajectory_absolute",
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    print(f"Saved finetuning data to {OUTPUT_PATH}")


if __name__ == "__main__":
    convert_all_to_absolute()
