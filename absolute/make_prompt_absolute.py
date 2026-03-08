import json
from pathlib import Path

# ----------------------------
# 1️⃣ 시스템 지시문
# ----------------------------
prompt_system = (
    "You are a motion prediction assistant that forecasts future absolute joint coordinates "
    "based on the observed motion sequence."
)

# {0}: 관찰 프레임 수, {1}: 예측 프레임 수, {2}: 데이터 본문
prompt_template = (
    "Forecast the next {1:d} absolute (x, y) coordinates for all observed joints using the given {0:d} observed frames.\n"
    "Each coordinate is tokenized as [NUM][DEC]000[SEP][DEC]00000[ENDNUM].\n"
    "Return a valid JSON object with predicted coordinates for each joint in the same format.\n"
    "{2:s}"
)

DEFAULT_SCALE_CONFIG = "../BaseLine/absolute_scaling_config.json"
DEC_DIGITS = 5
DEC_SCALE = 10**DEC_DIGITS

# ----------------------------
# 2️⃣ 수치형 토큰 변환
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

# ----------------------------
# 3️⃣ 프롬프트 생성 함수 (안전장치 삭제 버전)
# ----------------------------
def create_prompt_absolute(
    path: str,
    apply_absolute_scaling: bool = True,
    absolute_scale_config_path: str = DEFAULT_SCALE_CONFIG,
):
    """
    OpenPose JSON 파일 → 절대 좌표(x, y) 추출 → 팔 관절 필터링 → 50:50 분할 프롬프트 생성
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pose_seq = data["pose_sequence"]
    joint_axis_scales, eps, clip_enabled, clip_c = ({}, 1e-6, False, 0.0)
    if apply_absolute_scaling:
        joint_axis_scales, eps, clip_enabled, clip_c = load_scale_config(absolute_scale_config_path)

    ARM_JOINTS = [
        "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
        "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    ]

    # 1. 데이터 정제 및 필터링
    absolute_seq = []
    for frame in pose_seq:
        frame_coords = {}
        valid_frame = True
        for joint in ARM_JOINTS:
            if joint in frame:
                x, y = frame[joint]
                if apply_absolute_scaling:
                    x = scale_coord(x, joint, "x", joint_axis_scales, eps, clip_enabled, clip_c)
                    y = scale_coord(y, joint, "y", joint_axis_scales, eps, clip_enabled, clip_c)
                else:
                    x = round(x, DEC_DIGITS)
                    y = round(y, DEC_DIGITS)
                frame_coords[joint] = (x, y)
            else:
                valid_frame = False
                break
        if valid_frame:
            absolute_seq.append(frame_coords)

    # 🚨 [수정] 고정 길이(8프레임 등) 안전장치를 삭제하고 Delta 코드와 동일하게 50% 분할
    mid = len(absolute_seq) // 2
    obs_seq = absolute_seq[:mid]
    pred_len = len(absolute_seq) - mid

    # 2. 텍스트 포맷팅
    def format_sequence(seq):
        lines = []
        for f in seq:
            parts = [
                f"{joint}:({num_to_tokens(f[joint][0])},{num_to_tokens(f[joint][1])})"
                for joint in ARM_JOINTS
            ]
            lines.append(" | ".join(parts))
        return "\n".join(lines)

    obs_text = format_sequence(obs_seq)

    # 3. 최종 예측 프롬프트 조립
    # {0}: len(obs_seq), {1}: pred_len
    base_prompt = prompt_template.format(len(obs_seq), pred_len, obs_text)
    
    final_prompt = (
        f"{prompt_system}\n\n"
        f"### Instruction ###\n{base_prompt}\n### End Instruction ###"
    )

    return final_prompt

# 독립 테스트
if __name__ == "__main__":
    path = "./collected_motions/raise_arm.json"
    try:
        prompt = create_prompt_absolute(path)
        print("[OK] Absolute prompt created (variable length mode).")
        print(prompt[:1000]) # 앞부분 출력
    except FileNotFoundError:
        print(f"[ERROR] File not found: {path}")
