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
# 절대 좌표용으로 학습된 LoRA 가중치가 있다면 해당 경로로 수정 권장
base_model_dir = "../Meta-Llama-3.1-8B_tokenizerExtension"
lora_model_dir = "../BaseLine/motionQA_delta_finetuned_lora_mps" ## MPS에서 파인튜닝한 delta 모델

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
pose_json_path = "../dataset/armRaise/IMG_8646.json"
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

input_len = inputs["input_ids"].shape[1]
new_token_ids = outputs[0][input_len:]
generated_text = tokenizer.decode(new_token_ids, skip_special_tokens=False)
print("\n🧠 [Model Raw Output]\n", generated_text[:1000])

# ----------------------------
# 4️⃣ 수치 토큰 → float 변환 (동일)
# ----------------------------
def decode_token_number(token_str: str):
    # 학습 포맷: {sign}[NUM][INT]{int}[SEP][DEC]{dec}[ENDNUM]
    pattern = r"(-?)\[NUM\]\[INT\](\d{3})\[SEP\]\[DEC\](\d{3})\[ENDNUM\]"
    m = re.search(pattern, token_str)
    if not m:
        raise ValueError(f"Invalid token format: {token_str}")
    sign = -1 if m.group(1) == "-" else 1
    int_part = int(m.group(2))
    dec_part = int(m.group(3))
    return round(sign * (int_part + dec_part / 1000), 4)

# ----------------------------
# 5️⃣ LLM 출력 → joint별 절대 좌표 수집
# ----------------------------
def parse_predicted_coordinates(llm_output: str):
    # 학습 완료 포맷: JOINT:(tok,tok),(tok,tok),... | JOINT:...
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
            val_x = decode_token_number(x_tok)
            val_y = decode_token_number(y_tok)
            frames.append((val_x, val_y))
        if frames:
            result[joint] = frames
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

valid_connections = [
    (a, b) for (a, b) in MEDIAPIPE_CONNECTIONS
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
max_frames = max([len(seq) for seq in abs_positions.values()])

fig, ax = plt.subplots(figsize=(6, 6))

def update(frame):
    ax.clear()
    for jA, jB in valid_connections:
        xA, yA = abs_positions[jA][frame]
        xB, yB = abs_positions[jB][frame]

        ax.plot([xA, xB], [yA, yB], "-o", linewidth=3, markersize=6)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymax, ymin) # Y축 반전 (영상 좌표계)
    ax.grid(True)
    ax.set_title(f"Predicted Absolute Motion – Frame {frame}")

ani = FuncAnimation(fig, update, frames=max_frames, interval=150)

# 논문용 GIF 저장
# writer = PillowWriter(fps=10)
# ani.save("predicted_absolute_motion.gif", writer=writer)

plt.show()

# 1. JSON 저장 (예측된 절대 좌표)
# LLM이 예측한 delta를 baseline에 누적한 '최종 절대 좌표'를 저장합니다.
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
