import json
import re
from pathlib import Path

DEFAULT_SCALE_CONFIG = "../BaseLine/absolute_scaling_config.json"
MARKER_OBS = "Observed absolute coordinates:"
MARKER_NEXT = "Next absolute coordinates:"
DEC_DIGITS = 5
DEC_SCALE = 10**DEC_DIGITS

# ----------------------------
# 시스템 지시문 + 프롬프트 템플릿
# ----------------------------
PROMPT_SYSTEM = (
    "You are a motion prediction assistant that extrapolates future joint movements "
    "based on observed absolute (x, y) coordinate sequences. "
    "IMPORTANT: Provide EXACTLY 8 frames. Do not provide more or less."
)

PROMPT_TEMPLATE = (
    "Forecast the next {pred_len:d} (x, y) absolute coordinates for all observed joints "
    "using the given {obs_len:d} observed coordinate frames.\n"
    "Each coordinate value must follow this token format: -[NUM][INT]000[SEP][DEC]00000[ENDNUM] "
    "(minus sign optional).\n"
    "Return one line in this exact format:\n"
    "Next absolute coordinates: JOINT:(tok,tok),(tok,tok),... | JOINT:(tok,tok),...\n"
    "### Observed Absolute Coordinate Sequences ###\n"
    "{obs_text}"
)

ARM_JOINTS = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
]


def num_to_tokens(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    int_part = int(x)
    dec_part = int(round((x - int_part) * DEC_SCALE))
    if dec_part >= DEC_SCALE:
        int_part += 1
        dec_part -= DEC_SCALE
    return f"{sign}[NUM][INT]{int_part:03d}[SEP][DEC]{dec_part:0{DEC_DIGITS}d}[ENDNUM]"


def load_scale_config(scale_config_path: str):
    path = Path(scale_config_path)
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent / path).resolve()
    if not path.exists():
        return {}, 1e-6, False, 0.0
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    eps = float(payload.get("eps", 1e-6))
    clip_enabled = bool(payload.get("clip_enabled", False))
    clip_c = float(payload.get("clip_c", 0.0))
    scales = payload.get("joint_axis_scale", {})
    normalized = {}
    for joint, s in scales.items():
        if not isinstance(s, dict):
            continue
        sx = float(s.get("x", 1.0))
        sy = float(s.get("y", 1.0))
        if sx <= 0:
            sx = 1.0
        if sy <= 0:
            sy = 1.0
        normalized[str(joint)] = {"x": sx, "y": sy}
    return normalized, eps, clip_enabled, clip_c


def scale_coord(
    val: float,
    joint: str,
    axis: str,
    joint_axis_scales: dict[str, dict[str, float]],
    eps: float,
    clip_enabled: bool,
    clip_c: float,
) -> float:
    s = joint_axis_scales.get(joint, {}).get(axis, 1.0)
    out = val / (s + eps)
    if clip_enabled and clip_c > 0:
        out = max(-clip_c, min(clip_c, out))
    return round(out, DEC_DIGITS)


def create_prompt_absolute(
    path_or_data,
    pred_len: int = 8,
    max_obs: int = 10,
    with_system: bool = True,
    prompt_style: str = "train",
    apply_absolute_scaling: bool = True,
    absolute_scale_config_path: str = DEFAULT_SCALE_CONFIG,
):
    """
    Build absolute-coordinate prediction prompt from pose_sequence JSON.
    - train style: "Observed absolute coordinates: JOINT:(tok,tok),... | JOINT:..."
    - instruct style: system/template + observed section
    """
    if isinstance(path_or_data, (str, Path)):
        with open(path_or_data, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = path_or_data

    pose_seq = data.get("pose_sequence", [])
    if not pose_seq:
        return MARKER_OBS

    joint_axis_scales = {}
    eps = 1e-6
    clip_enabled = False
    clip_c = 0.0
    if apply_absolute_scaling:
        joint_axis_scales, eps, clip_enabled, clip_c = load_scale_config(absolute_scale_config_path)

    joints_list = [j for j in pose_seq[0].keys() if j in ARM_JOINTS]
    if not joints_list:
        return MARKER_OBS

    obs_seq = pose_seq[-max_obs:] if max_obs > 0 else pose_seq
    parts = []
    obs_len = 0
    for joint in joints_list:
        coords = []
        for frame in obs_seq:
            if joint not in frame:
                continue
            x, y = frame[joint]
            if apply_absolute_scaling:
                x = scale_coord(x, joint, "x", joint_axis_scales, eps, clip_enabled, clip_c)
                y = scale_coord(y, joint, "y", joint_axis_scales, eps, clip_enabled, clip_c)
            else:
                x = round(x, DEC_DIGITS)
                y = round(y, DEC_DIGITS)
            coords.append(f"({num_to_tokens(x)},{num_to_tokens(y)})")
        if coords:
            obs_len = max(obs_len, len(coords))
            parts.append(f"{joint}:{','.join(coords)}")

    obs_text = " | ".join(parts)
    if not obs_text:
        return MARKER_OBS
    if prompt_style == "train":
        return f"{MARKER_OBS} {obs_text}"

    base_prompt = PROMPT_TEMPLATE.format(pred_len=pred_len, obs_len=obs_len, obs_text=obs_text)
    if with_system:
        return f"{PROMPT_SYSTEM}\n\n{base_prompt}"
    return base_prompt


def build_instruct_prompt_from_observed(
    observed_prompt: str,
    pred_len: int = 8,
    with_system: bool = True,
) -> str:
    """
    Convert JSONL train-style observed prompt to instruct-style prompt.
    Input example: "Observed absolute coordinates: JOINT:(tok,tok),... | JOINT:..."
    """
    text = (observed_prompt or "").strip()
    prefix = MARKER_OBS
    obs_text = text[len(prefix) :].strip() if text.startswith(prefix) else text

    obs_len = 0
    if obs_text:
        first_seg = obs_text.split(" | ", 1)[0]
        if ":" in first_seg:
            first_seg = first_seg.split(":", 1)[1]
        frame_pat = re.compile(
            r"\(\s*-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{5}\[ENDNUM\]\s*,\s*"
            r"-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{5}\[ENDNUM\]\s*\)"
        )
        obs_len = len(frame_pat.findall(first_seg))

    base_prompt = PROMPT_TEMPLATE.format(pred_len=pred_len, obs_len=obs_len, obs_text=obs_text)
    if with_system:
        return f"{PROMPT_SYSTEM}\n\n{base_prompt}"
    return base_prompt


if __name__ == "__main__":
    path = "./collected_motions/raise_arm.json"
    try:
        prompt = create_prompt_absolute(path)
        print("[OK] Absolute train-style prompt created.")
        print(prompt[:1000])
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
