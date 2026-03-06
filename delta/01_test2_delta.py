import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from make_prompt_from_pose import create_prompt_from_pose
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# 이 스크립트는 다음 순서로 동작한다.
# 1) 베이스 모델 + LoRA 로드
# 2) pose JSON으로 프롬프트 생성
# 3) 생성 결과에서 delta 토큰 파싱
# 4) delta를 절대좌표로 누적해 JSON/GIF로 저장

# ----------------------------
# 1️⃣ 모델 & 토크나이저 로드
# ----------------------------
base_model_dir = "../Meta-Llama-3.1-8B_tokenizerExtension"
lora_model_dir = "../BaseLine/motionQA_delta_finetuned_lora_mps" ## MPS에서 파인튜닝한 모델

print("🔹 Loading model...")
tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(base_model_dir, local_files_only=True)
model = PeftModel.from_pretrained(model, lora_model_dir)
model.eval()

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).to(torch.float32)
print("✅ Model + LoRA Adapter Loaded Successfully!")

# ----------------------------
# 2️⃣ 프롬프트 생성
# ----------------------------
# 여기서 MP4 파일을 Json 파일로 변환한 델타 데이터를 삽입
pose_json_path = "../dataset/armRaise/IMG_8646.json"
test_prompt = create_prompt_from_pose(pose_json_path)

print("\n🧩 [Generated Test Prompt Preview]")
print(test_prompt[:600])

# ----------------------------
# 3️⃣ 모델 추론
# ----------------------------
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=15000,
        temperature=0.4,
        top_p=0.9,
        do_sample=True,
    )

input_len = inputs["input_ids"].shape[1]
new_token_ids = outputs[0][input_len:]
generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
print("\n🧠 [Model Raw Output]\n", generated_text[:1000])


# ----------------------------
# 4️⃣ delta 토큰 → float 변환
# ----------------------------
def decode_token_number(token_str: str):
    # 학습 포맷: {sign}[NUM][INT]{int}[SEP][DEC]{dec}[ENDNUM]
    # 부호가 [NUM] 앞에 위치, 정수부는 [INT] 토큰 사용
    pattern = r"(-?)\[NUM\]\[INT\](\d{3})\[SEP\]\[DEC\](\d{3})\[ENDNUM\]"
    m = re.search(pattern, token_str)
    if not m:
        raise ValueError(f"Invalid token format: {token_str}")
    sign = -1 if m.group(1) == "-" else 1
    int_part = int(m.group(2))
    dec_part = int(m.group(3))
    return round(sign * (int_part + dec_part / 1000), 4)


# ----------------------------
# 5️⃣ LLM 출력 → joint별 delta 수집
# ----------------------------
def parse_predicted_deltas(llm_output: str):
    # 학습 완료 포맷: "Next motion deltas: JOINT:(tok,tok),(tok,tok),... | JOINT:..."
    # "Next motion deltas:" 이후 텍스트만 사용
    marker = "Next motion deltas:"
    start = llm_output.find(marker)
    if start != -1:
        llm_output = llm_output[start + len(marker):]

    frame_pat = re.compile(
        r"\((-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{3}\[ENDNUM\])"
        r",(-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{3}\[ENDNUM\])\)"
    )
    joint_pat = re.compile(r"([A-Z][A-Z_]*):")

    result = {}
    for seg in llm_output.split(" | "):
        jm = joint_pat.match(seg.strip())
        if not jm:
            continue
        joint = jm.group(1)
        frames = []
        for x_tok, y_tok in frame_pat.findall(seg):
            dx = decode_token_number(x_tok)
            dy = decode_token_number(y_tok)
            frames.append((dx, dy))
        if frames:
            result[joint] = frames
    return result


pred_deltas = parse_predicted_deltas(generated_text)

if not pred_deltas:
    print("\n❌ [PARSE FAILED] 모델 출력에서 delta 토큰을 파싱하지 못했습니다.")
    print("Raw output (first 500 chars):", repr(generated_text[:500]))
    raise SystemExit(1)

# ----------------------------
# 6️⃣ 🔥 최종 출력: delta만 출력
# ----------------------------
print("\n==============================")
print("🔍 LLM Predicted Delta Sequences (float only)")
print("==============================")

for joint, seq in pred_deltas.items():
    print(f"\n🦴 {joint} ({len(seq)} frames predicted)")
    for i, (dx, dy) in enumerate(seq):
        print(f" [{i}]  dx={dx}, dy={dy}")


# 🔥 [수정] max_frames 계산 코드 추가
# 첫 번째 관절의 예측 길이를 기준으로 설정
frame_counts = {j: len(seq) for j, seq in pred_deltas.items()}
max_frames = min(frame_counts.values())

print(f"📊 Predicted Frames (min): {max_frames}")
print(f"📊 Joints parsed: {len(pred_deltas)}")
print(f"📊 Frame counts per joint: {frame_counts}")

# ===========================================================
# 7️⃣ 🔥 Skeleton 2D Animation (FIXED & CONSISTENT)
# ... (이하 코드는 그대로 유지)
# ===========================================================
# 7️⃣ 🔥 Skeleton 2D Animation (FIXED & CONSISTENT)
# ===========================================================
with open(pose_json_path, "r") as f:
    pose_data = json.load(f)

pose_sequence = pose_data["pose_sequence"]
baseline = pose_sequence[-1]  # 마지막 GT 프레임

# ----------------------------
# MediaPipe full connections
# ----------------------------
MEDIAPIPE_CONNECTIONS = [
    ("NOSE", "LEFT_EYE_INNER"),
    ("LEFT_EYE_INNER", "LEFT_EYE"),
    ("LEFT_EYE", "LEFT_EYE_OUTER"),
    ("NOSE", "RIGHT_EYE_INNER"),
    ("RIGHT_EYE_INNER", "RIGHT_EYE"),
    ("RIGHT_EYE", "RIGHT_EYE_OUTER"),
    ("LEFT_EYE_OUTER", "LEFT_EAR"),
    ("RIGHT_EYE_OUTER", "RIGHT_EAR"),
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_WRIST", "LEFT_THUMB"),
    ("LEFT_WRIST", "LEFT_INDEX"),
    ("LEFT_WRIST", "LEFT_PINKY"),
    ("RIGHT_WRIST", "RIGHT_THUMB"),
    ("RIGHT_WRIST", "RIGHT_INDEX"),
    ("RIGHT_WRIST", "RIGHT_PINKY"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("LEFT_ANKLE", "LEFT_HEEL"),
    ("LEFT_ANKLE", "LEFT_FOOT_INDEX"),
    ("RIGHT_ANKLE", "RIGHT_HEEL"),
    ("RIGHT_ANKLE", "RIGHT_FOOT_INDEX"),
]


# ----------------------------
# delta → absolute 누적
# ----------------------------
def accumulate(base, deltas):
    # 기준 좌표(base)에서 delta를 누적해 절대좌표 시퀀스로 변환
    x, y = base
    coords = [(x, y)]
    for dx, dy in deltas:
        x += dx
        y += dy
        coords.append((x, y))
    return coords


abs_positions = {
    j: accumulate(baseline[j], seq)
    for j, seq in pred_deltas.items()
    if j in baseline  # baseline에 없는 joint는 건너뜀 (미관측 프레임 대비)
}

print("[DEBUG] abs_positions joints:", list(abs_positions.keys()))

# ----------------------------
# 실제 존재하는 관절만 connection 필터링
# ----------------------------
available_joints = set(abs_positions.keys())

valid_connections = [
    (a, b)
    for (a, b) in MEDIAPIPE_CONNECTIONS
    if a in available_joints and b in available_joints
]

print("[DEBUG] valid connections:", valid_connections)

# ----------------------------
# axis 자동 계산 (중요)
# ----------------------------
all_x, all_y = [], []
for seq in abs_positions.values():
    for x, y in seq:
        all_x.append(x)
        all_y.append(y)

xmin, xmax = min(all_x) - 0.05, max(all_x) + 0.05
ymin, ymax = min(all_y) - 0.05, max(all_y) + 0.05

# ===========================================================
# 8️⃣ 🔥 결과 저장 (JSON & GIF) - [Delta 방식 최종본]
# ===========================================================

# --- [A] JSON 저장 (누적된 절대 좌표) ---
# Delta 모델의 예측값을 실제 좌표로 변환한 최종 데이터를 저장합니다.
output_json_filename = "predicted_motion_delta_result.json"
final_pose_sequence = []

# Dictionary of lists (abs_positions) -> List of dictionaries (Frame-wise) 변환
for i in range(max_frames):
    frame_data = {}
    for joint, coords in abs_positions.items():
        frame_data[joint] = coords[i]
    final_pose_sequence.append(frame_data)

with open(output_json_filename, "w", encoding="utf-8") as f:
    json.dump({"pose_sequence": final_pose_sequence}, f, indent=4, ensure_ascii=False)
print(f"\n💾 Predicted Delta motion saved to JSON: {output_json_filename}")


# --- [B] 애니메이션 설정 ---
fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    # 프레임별 스켈레톤 렌더링 콜백
    ax.clear()
    # 유효한 연결선 그리기
    for jA, jB in valid_connections:
        xA, yA = abs_positions[jA][frame]
        xB, yB = abs_positions[jB][frame]

        # Delta 방식은 초록색 계열로 그려 Absolute(파란색)와 시각적으로 구분합니다.
        ax.plot([xA, xB], [yA, yB], "-o", linewidth=3, markersize=6, color='seagreen')

    # 축 및 그리드 설정
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin)  # Y축 반전 (영상 좌표계 일치)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.set_title(f"Predicted Delta Motion – Frame {frame}")

# 애니메이션 객체 생성 (200ms 간격)
ani = FuncAnimation(fig, update, frames=max_frames, interval=200)


# --- [C] GIF 저장 (plt.show() 호출 전에 수행해야 안전함) ---
output_gif_filename = "predicted_motion_delta.gif"
print(f"💾 Saving animation to {output_gif_filename} (Please wait)...")

# fps=10으로 설정하여 Absolute 결과와 속도를 맞춥니다.
ani.save(output_gif_filename, writer=PillowWriter(fps=10))
print(f"✅ Delta Animation saved successfully: {output_gif_filename}")


# --- [D] 화면 표시 ---
plt.show()
