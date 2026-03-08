import json
from pathlib import Path

# 이 모듈은 pose_sequence(JSON)를 읽어
# "관측 delta -> 미래 delta 예측" 프롬프트 문자열을 생성한다.
# 학습 데이터(finetune_dataset_delta_*.jsonl)와 동일한 포맷으로 생성한다.

# ----------------------------
# 시스템 지시문 + 프롬프트 템플릿
# ----------------------------
PROMPT_SYSTEM = (
    "You are a motion prediction assistant that extrapolates future joint movements "
    "based on observed delta (dx, dy) coordinate sequences. "
    "IMPORTANT: Provide EXACTLY 8 frames. Do not provide more or less."
)

PROMPT_TEMPLATE = (
    "Forecast the next {pred_len:d} (x, y) coordinate deltas for all observed joints "
    "using the given {obs_len:d} observed delta frames.\n"
    "Each delta value must follow this token format: -[NUM][INT]000[SEP][DEC]00000[ENDNUM] "
    "(minus sign optional).\n"
    "Return one line in this exact format:\n"
    "Next motion deltas: JOINT:(tok,tok),(tok,tok),... | JOINT:(tok,tok),...\n"
    "### Observed Delta Sequences ###\n"
    "{obs_text}"
)

ARM_JOINTS = {
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
}

DEFAULT_SCALE_CONFIG = "../BaseLine/delta_scaling_config.json"
DEC_DIGITS = 5
DEC_SCALE = 10**DEC_DIGITS

# ----------------------------
# 수치형 토큰 변환 (학습 포맷: {sign}[NUM][INT]{int:03d}[SEP][DEC]{dec:05d}[ENDNUM])
# ----------------------------
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
    scales = payload.get("joint_scale", {})
    if not scales:
        # backward compatibility: if only joint_gain exists, invert it.
        gains = payload.get("joint_gain", {})
        for k, v in gains.items():
            if isinstance(v, (int, float)) and v > 0:
                scales[str(k)] = (1.0 / float(v))
    scales = {
        str(k): float(v)
        for k, v in scales.items()
        if isinstance(v, (int, float)) and v > 0
    }
    return scales, eps, clip_enabled, clip_c


def scale_delta_value(
    value: float,
    joint: str,
    joint_scales: dict[str, float],
    eps: float,
    clip_enabled: bool,
    clip_c: float,
) -> float:
    s = joint_scales.get(joint, 1.0)
    out = value / (s + eps)
    if clip_enabled and clip_c > 0:
        out = max(-clip_c, min(clip_c, out))
    return round(out, DEC_DIGITS)


# ----------------------------
# 프롬프트 생성 함수
# ----------------------------
def create_prompt_from_pose(
    path_or_data,
    pred_len: int = 8,
    max_obs: int = 10,
    with_system: bool = True,
    prompt_style: str = "train",
    apply_delta_scaling: bool = True,
    delta_scale_config_path: str = DEFAULT_SCALE_CONFIG,
):
    """
    JSON 파일 경로 또는 데이터 객체 -> 최근 10개 delta 프레임 추출
    -> 학습 데이터와 동일한 joint-first 포맷 프롬프트 생성
    포맷: "Observed motion deltas: JOINT:(tok,tok),(tok,tok),... | JOINT:..."
    """
    if isinstance(path_or_data, str):
        with open(path_or_data, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = path_or_data

    joint_scales = {}
    scale_eps = 1e-6
    clip_enabled = False
    clip_c = 0.0
    if apply_delta_scaling:
        joint_scales, scale_eps, clip_enabled, clip_c = load_scale_config(delta_scale_config_path)

    pose_seq = data["pose_sequence"]
    if not pose_seq:
        return "Observed motion deltas: "

    joints_list = [j for j in pose_seq[0].keys() if j in ARM_JOINTS]
    joint_deltas = {j: [] for j in joints_list}
    for i in range(1, len(pose_seq)):
        if not pose_seq[i - 1] or not pose_seq[i]:
            continue
        for joint in joints_list:
            if joint in pose_seq[i - 1] and joint in pose_seq[i]:
                x1, y1 = pose_seq[i - 1][joint]
                x2, y2 = pose_seq[i][joint]
                dx = scale_delta_value(
                    x2 - x1, joint, joint_scales, scale_eps, clip_enabled, clip_c
                )
                dy = scale_delta_value(
                    y2 - y1, joint, joint_scales, scale_eps, clip_enabled, clip_c
                )
                joint_deltas[joint].append((dx, dy))

    obs = {j: joint_deltas[j][-max_obs:] for j in joints_list if joint_deltas[j]}
    parts = []
    for joint, frames in obs.items():
        frame_strs = [f"({num_to_tokens(dx)},{num_to_tokens(dy)})" for dx, dy in frames]
        parts.append(f"{joint}:{','.join(frame_strs)}")
    obs_text = " | ".join(parts)

    if prompt_style == "train":
        return f"Observed motion deltas: {obs_text}"

    obs_len = max((len(v) for v in obs.values()), default=0)
    base_prompt = PROMPT_TEMPLATE.format(pred_len=pred_len, obs_len=obs_len, obs_text=obs_text)
    if with_system:
        return f"{PROMPT_SYSTEM}\n\n{base_prompt}"
    return base_prompt
