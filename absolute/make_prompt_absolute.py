import json

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
    "Each coordinate is tokenized as [NUM][DEC]000[SEP][DEC]000[ENDNUM].\n"
    "Return a valid JSON object with predicted coordinates for each joint in the same format.\n"
    "{2:s}"
)

# ----------------------------
# 2️⃣ 수치형 토큰 변환
# ----------------------------
def num_to_tokens(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    int_part = int(x)
    dec_part = int(round((x - int_part) * 1000))
    return f"{sign}[NUM][INT]{int_part:03d}[SEP][DEC]{dec_part:03d}[ENDNUM]"

# ----------------------------
# 3️⃣ 프롬프트 생성 함수 (안전장치 삭제 버전)
# ----------------------------
def create_prompt_absolute(path: str):
    """
    OpenPose JSON 파일 → 절대 좌표(x, y) 추출 → 팔 관절 필터링 → 50:50 분할 프롬프트 생성
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    pose_seq = data["pose_sequence"]

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
                frame_coords[joint] = (round(x, 4), round(y, 4))
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
        print("✅ [Absolute Prompt Created (Variable Length Mode)]")
        print(prompt[:1000]) # 앞부분 출력
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {path}")