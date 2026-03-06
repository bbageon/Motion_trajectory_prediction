import json

# 이 모듈은 pose_sequence(JSON)를 읽어
# "관측 delta -> 미래 delta 예측" 프롬프트 문자열을 생성한다.
# 학습 데이터(finetune_dataset_delta_*.jsonl)와 동일한 포맷으로 생성한다.

# ----------------------------
# 수치형 토큰 변환 (학습 포맷: {sign}[NUM][INT]{int:03d}[SEP][DEC]{dec:03d}[ENDNUM])
# ----------------------------
def num_to_tokens(x: float) -> str:
    sign = "-" if x < 0 else ""
    x = abs(x)
    int_part = int(x)
    dec_part = int(round((x - int_part) * 1000))
    return f"{sign}[NUM][INT]{int_part:03d}[SEP][DEC]{dec_part:03d}[ENDNUM]"


# ----------------------------
# 프롬프트 생성 함수
# ----------------------------
def create_prompt_from_pose(path_or_data):
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

    pose_seq = data["pose_sequence"]

    joints_list = list(pose_seq[0].keys())  # 모든 joint 사용 (학습 데이터와 동일)

    # 1. 전체 시퀀스에서 Delta(변화량) 계산 (joint별로 누적)
    joint_deltas = {j: [] for j in joints_list}
    for i in range(1, len(pose_seq)):
        if not pose_seq[i - 1] or not pose_seq[i]:
            continue
        for joint in joints_list:
            if joint in pose_seq[i - 1] and joint in pose_seq[i]:
                x1, y1 = pose_seq[i - 1][joint]
                x2, y2 = pose_seq[i][joint]
                dx = round(x2 - x1, 3)
                dy = round(y2 - y1, 3)
                joint_deltas[joint].append((dx, dy))

    # 2. 최근 10프레임만 사용
    MAX_OBS = 10
    obs = {j: joint_deltas[j][-MAX_OBS:] for j in joints_list if joint_deltas[j]}

    # 3. joint-first 포맷으로 직렬화: JOINT:(tok,tok),(tok,tok),...
    parts = []
    for joint, frames in obs.items():
        frame_strs = [
            f"({num_to_tokens(dx)},{num_to_tokens(dy)})"
            for dx, dy in frames
        ]
        parts.append(f"{joint}:{','.join(frame_strs)}")

    obs_text = " | ".join(parts)

    # 4. 학습 포맷 그대로: "Observed motion deltas: ..."
    return f"Observed motion deltas: {obs_text}"
