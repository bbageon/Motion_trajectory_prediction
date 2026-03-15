import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from make_prompt_absolute import create_prompt_absolute 

# ----------------------------
# 1️⃣ 모델 & 토크나이저 로드
# ----------------------------
# 절대 좌표용으로 학습된 LoRA 가중치가 있다면 해당 경로로 수정하세요.
base_model_dir = "./llama2_local"
lora_model_dir = "./motionQA_finetuned_lora_mps"

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
# 2️⃣ 프롬프트 생성 (절대 좌표용 함수가 따로 있다면 교체)
# ----------------------------
pose_json_path = "./collected_motions/raise_arm_1764194988.json"
# create_prompt_from_pose 함수 내부에서 delta가 아닌 절대값을 뱉도록 구성되어 있어야 합니다.

test_prompt = create_prompt_absolute(pose_json_path)

print("\n🧩 [Generated Test Prompt Preview]")
print(test_prompt[:600])

# ----------------------------
# 3️⃣ 모델 추론
# ----------------------------
inputs = tokenizer(test_prompt, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=500, # 절대 좌표는 토큰 길이가 더 길 수 있으므로 여유있게 설정
        temperature=0.2,    # 논문용 실험 시 변동성을 줄이기 위해 낮춤
        top_p=0.9,
        do_sample=True,
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\n🧠 [Model Raw Output]\n", generated_text[:1000])

# ----------------------------
# 4️⃣ 수치 토큰 → float 변환 (동일)
# ----------------------------
def decode_token_number(token_str: str):
    pattern = r"\[NUM\](-?)(?:\[DEC\])?(\d{3})\[SEP\]\[DEC\](\d{3})\[ENDNUM\]"
    m = re.search(pattern, token_str)
    if not m:
        return None
    sign = -1 if m.group(1) == "-" else 1
    int_part = int(m.group(2))
    dec_part = int(m.group(3))
    v = round(sign * (int_part + dec_part / 1000), 4)
    return v

# ----------------------------
# 5️⃣ LLM 출력 → joint별 절대 좌표 수집
# ----------------------------
def parse_predicted_coordinates(llm_output: str):
    # 포맷: JOINT:( [NUM]... , [NUM]... )
    pattern = r"([A-Z_]+):\(\s*(\[NUM\].*?\[ENDNUM\])\s*,\s*(\[NUM\].*?\[ENDNUM\])\s*\)"
    matches = re.findall(pattern, llm_output)

    result = {}
    for joint, x_tok, y_tok in matches:
        val_x = decode_token_number(x_tok)
        val_y = decode_token_number(y_tok)

        if val_x is not None and val_y is not None:
            if joint not in result:
                result[joint] = []
            result[joint].append((val_x, val_y))
    return result

pred_coords = parse_predicted_coordinates(generated_text)

# ----------------------------
# 6️⃣ 🔥 데이터 변환: 절대 좌표이므로 누적 불필요
# ----------------------------
# 모델이 출력한 좌표를 그대로 사용합니다.
abs_positions = pred_coords

print("\n==============================")
print("🔍 LLM Predicted Absolute Coordinates (float only)")
print("==============================")
for joint, seq in abs_positions.items():
    print(f"🦴 {joint}: {len(seq)} frames predicted. First frame: {seq[0]}")

# ----------------------------
# 7️⃣ 시각화 준비 (Connections)
# ----------------------------
MEDIAPIPE_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"), ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"), ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"), ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"), ("LEFT_HIP", "RIGHT_HIP")
    # ... 필요한 연결선을 추가하세요
]

available_joints = set(abs_positions.keys())
valid_connections = [
    (a, b) for (a, b) in MEDIAPIPE_CONNECTIONS
    if a in available_joints and b in available_joints
]

# 축 범위 설정
all_x, all_y = [], []
for seq in abs_positions.values():
    for x, y in seq:
        all_x.append(x); all_y.append(y)

xmin, xmax = min(all_x) - 0.1, max(all_x) + 0.1
ymin, ymax = min(all_y) - 0.1, max(all_y) + 0.1

# ----------------------------
# 8️⃣ Animation & Save
# ----------------------------
max_frames = max([len(seq) for seq in abs_positions.values()]) if abs_positions else 0

fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    for jA, jB in valid_connections:
        # 프레임 인덱스 초과 방지
        idxA = min(frame, len(abs_positions[jA]) - 1)
        idxB = min(frame, len(abs_positions[jB]) - 1)

        xA, yA = abs_positions[jA][idxA]
        xB, yB = abs_positions[jB][idxB]

        ax.plot([xA, xB], [yA, yB], "-o", linewidth=3, markersize=6)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin) # Y축 반전 (영상 좌표계)
    ax.grid(True)
    ax.set_title(f"Predicted Absolute Motion – Frame {frame}")

if max_frames > 0:
    ani = FuncAnimation(fig, update, frames=max_frames, interval=150)
    
    # 논문용 GIF 저장
    # writer = PillowWriter(fps=10)
    # ani.save("predicted_absolute_motion.gif", writer=writer)
    
    plt.show()
else:
    print("❌ No valid predicted coordinates found.")

# 1. JSON 저장 (예측된 절대 좌표)
# LLM이 예측한 delta를 baseline에 누적한 '최종 절대 좌표'를 저장합니다.
output_json_filename = "predicted_motion_delta_result.json"
final_pose_sequence = []

# Dictionary of lists (abs_positions) -> List of dictionaries (Frame-wise) 변환
for i in range(max_frames):
    frame_data = {}
    for joint, coords in abs_positions.items():
        if i < len(coords):
            frame_data[joint] = coords[i]
    final_pose_sequence.append(frame_data)

with open(output_json_filename, "w", encoding="utf-8") as f:
    json.dump({"pose_sequence": final_pose_sequence}, f, indent=4, ensure_ascii=False)

print(f"\n💾 Predicted motion saved to JSON: {output_json_filename}")


# 2. 애니메이션 저장 (GIF)
# PillowWriter를 사용하여 GIF로 저장합니다.
output_gif_filename = "predicted_motion_delta.gif"
print(f"💾 Saving animation to {output_gif_filename} (This might take a moment)...")

# fps=5~10 정도로 설정하는 것이 논문용 시각화에 적당합니다.
ani.save(output_gif_filename, writer=PillowWriter(fps=10))
print("✅ Animation saved successfully!")

# 화면에 애니메이션 표시
plt.show()