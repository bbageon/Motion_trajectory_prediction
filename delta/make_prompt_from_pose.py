import json

# 이 모듈은 pose_sequence(JSON)를 읽어
# "관측 delta -> 미래 delta 예측" 프롬프트 문자열을 생성한다.

# ----------------------------
# 1️⃣ 시스템 지시문 (예측 프레임 수 명시)
# ----------------------------
prompt_system = (
    "You are a motion prediction assistant that extrapolates future joint movements "
    "based on observed delta (Δx, Δy) coordinate sequences."
    "IMPORTANT: Provide EXACTLY 8 frames. Do not provide more or less."
)


# prompt_system = (
#     "You are a motion prediction assistant that extrapolates future joint movements "
#     "based on observed delta (Δx, Δy) coordinate sequences."
# )

# {1:d} 부분에 강제로 8이 들어가도록 구성됩니다.
prompt_template = (
    "Forecast the next {1:d} (x, y) coordinate deltas for all observed joints using the given {0:d} observed delta frames.\n"
    "Each delta value is tokenized as [NUM][DEC]000[SEP][DEC]000[ENDNUM].\n"
    "Return a valid JSON object with predicted deltas for each joint in the same format.\n"
    "### Observed Delta Sequences ###\n"
    "{2:s}"
)


# ----------------------------
# 2️⃣ 수치형 토큰 변환
# ----------------------------
def num_to_tokens(x: float) -> str:
    # 실수값을 고정 포맷 토큰으로 변환한다.
    # 예: -0.012 -> [NUM]-[DEC]000[SEP][DEC]012[ENDNUM]
    sign = "-" if x < 0 else ""
    x = abs(x)
    int_part = int(x)
    dec_part = int(round((x - int_part) * 1000))
    return f"[NUM]{sign}[DEC]{int_part:03d}[SEP][DEC]{dec_part:03d}[ENDNUM]"


# ----------------------------
# 3️⃣ 프롬프트 생성 함수 (입력 10, 예측 8 제한)
# ----------------------------
def create_prompt_from_pose(path_or_data):
    """
    JSON 파일 경로 또는 데이터 객체 -> 최근 10개 프레임 추출 -> 8개 예측 프롬프트 생성
    """
    # 데이터 로드 (경로일 경우와 dict일 경우 모두 대응)
    if isinstance(path_or_data, str):
        with open(path_or_data, encoding="utf-8") as f:
            data = json.load(f)
    else:
        # 이미 dict 형태로 전달된 경우
        data = path_or_data

    pose_seq = data["pose_sequence"]

    # LLM이 집중할 주요 관절 (필요에 따라 조절)
    ARM_JOINTS = [
        "RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST",
        "LEFT_SHOULDER", "LEFT_ELBOW", "LEFT_WRIST",
    ]

    # 1. 전체 시퀀스에서 Delta(변화량) 계산
    all_deltas = []
    for i in range(1, len(pose_seq)):
        frame_delta = {}
        # 이전 프레임과 현재 프레임 모두 데이터가 있는 경우만 계산
        if not pose_seq[i-1] or not pose_seq[i]:
            continue
            
        for joint in ARM_JOINTS:
            if joint in pose_seq[i-1] and joint in pose_seq[i]:
                x1, y1 = pose_seq[i-1][joint]
                x2, y2 = pose_seq[i][joint]
                dx, dy = round(x2 - x1, 4), round(y2 - y1, 4)
                frame_delta[joint] = (dx, dy)
        
        if frame_delta:
            all_deltas.append(frame_delta)

    # 2. ⭐ 입력 프레임 제한 (최근 10개만 사용)
    obs_deltas = all_deltas[-10:] 
    obs_len = len(obs_deltas)
    
    # 3. ⭐ 예측 프레임 고정 (8프레임)
    pred_len = 8

    # 4. Delta 데이터를 텍스트 토큰으로 변환
    def format_deltas(seq):
        # 프레임별로 "JOINT:(dx_token,dy_token)" 문자열을 만든다.
        lines = []
        for i, f in enumerate(seq):
            parts = [
                f"{joint}:({num_to_tokens(dx)},{num_to_tokens(dy)})"
                for joint, (dx, dy) in f.items()
            ]
            lines.append(f"Frame[{i}]: " + " | ".join(parts))
        return "\n".join(lines)

    obs_text = format_deltas(obs_deltas)

    # 5. 최종 프롬프트 조립
    base_prompt = prompt_template.format(obs_len, pred_len, obs_text)
    final_prompt = (
        f"{prompt_system}\n\n"
        f"### Instruction ###\n{base_prompt}\n### End Instruction ###\n"
        f"Answer:"
    )

    return final_prompt
