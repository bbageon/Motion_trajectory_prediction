import json
import torch
import re
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =================================================================
# 1️⃣ 설정
# =================================================================
# 🚨 경로 확인 필수!
base_model_dir = "./llama2_local"
lora_model_dir = "./motionQA_delta_finetuned_lora_5080_scaling" 
test_file_path = "./TestData/Squart_Test_data.json"
output_json_path = "./prediction_result_final.json"

SCALE_FACTOR = 1000.0   
VELOCITY_LIMIT = 50.0   
FLOOR_Y = 1000.0        
MOVEMENT_THRESHOLD = 2.0 

print("🔹 Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# =================================================================
# 2️⃣ 모델 로드 (수정됨!)
# =================================================================
# 🚨 [수정] device 변수 정의 추가
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(base_model_dir, local_files_only=True)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    base_model_dir, 
    local_files_only=True,
    quantization_config=bnb_config, 
    device_map="auto"
)

model = PeftModel.from_pretrained(model, lora_model_dir)
model.eval()
print(f"✅ Model Loaded on {device}!")

# =================================================================
# 3️⃣ 유틸리티 함수
# =================================================================
def num_to_tokens(val: float) -> str:
    sign = "-" if val < 0 else ""
    val = abs(val)
    int_part = int(val)
    dec_part = int(round((val - int_part) * 1000))
    return f"[NUM]{sign}[DEC]{str(int_part).zfill(3)}[SEP][DEC]{str(dec_part).zfill(3)}[ENDNUM]"

def decode_token_number(token_str: str):
    pattern = r"\[NUM\](-?)(?:\[DEC\])?(\d{3})\[SEP\]\[DEC\](\d{3})\[ENDNUM\]"
    m = re.search(pattern, token_str)
    if not m: return 0.0
    sign = -1 if m.group(1) == "-" else 1
    v = round(sign * (int(m.group(2)) + int(m.group(3)) / 1000), 4)
    return v

def parse_one_frame(llm_output: str):
    result = {}
    joint_chunks = llm_output.split('|')
    for chunk in joint_chunks:
        if ':' not in chunk: continue
        try:
            joint_name, data_part = chunk.split(':', 1)
            joint_name = joint_name.strip()
            tuple_matches = re.findall(r"\((.*?)\)", data_part)
            if tuple_matches:
                tm = tuple_matches[0]
                if ',' in tm:
                    x_str, y_str = tm.split(',', 1)
                    dx = decode_token_number(x_str.strip())
                    dy = decode_token_number(y_str.strip())
                    result[joint_name] = (dx, dy)
        except:
            continue
    return result

# =================================================================
# 4️⃣ 데이터 준비
# =================================================================
with open(test_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

full_pose_seq = data["pose_sequence"]
TARGET_JOINTS = [
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW",
    "LEFT_WRIST", "RIGHT_WRIST", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE"
]

input_context_len = 5
current_seq = []

print(f"📊 Reading last {input_context_len} frames (Scaled x{SCALE_FACTOR})...")
for frame in full_pose_seq[-input_context_len:]:
    cleaned = {}
    for k in TARGET_JOINTS:
        if k in frame:
            x_raw, y_raw = frame[k]
            cleaned[k] = [x_raw * SCALE_FACTOR, y_raw * SCALE_FACTOR]
    current_seq.append(cleaned)

start_frame = current_seq[-1]
PREDICT_COUNT = 8
predicted_frames = []

print(f"\n🚀 Forecasting next {PREDICT_COUNT} frames...")
print("=" * 60)

# =================================================================
# 5️⃣ 예측 루프
# =================================================================
for step in range(PREDICT_COUNT):
    last_frame = current_seq[-1]
    prev_frame = current_seq[-2]
    
    formatted_joints = []
    for joint in TARGET_JOINTS:
        if joint in last_frame and joint in prev_frame:
            dx = round(last_frame[joint][0] - prev_frame[joint][0], 3)
            dy = round(last_frame[joint][1] - prev_frame[joint][1], 3)
            dx_tok = num_to_tokens(dx)
            dy_tok = num_to_tokens(dy)
            formatted_joints.append(f"{joint}:({dx_tok},{dy_tok})")
    
    if not formatted_joints:
         obs_str = " | ".join([f"{j}:({num_to_tokens(0.0)},{num_to_tokens(0.0)})" for j in TARGET_JOINTS])
    else:
        obs_str = " | ".join(formatted_joints)

    prompt = f"Observed motion deltas: {obs_str}\nNext motion deltas:"
    
    # 🚨 [수정] 이제 device가 정의되어 있으므로 에러가 나지 않습니다.
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
            repetition_penalty=1.1
        )
    
    gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "Next motion deltas:" in gen_text:
        pred_text = gen_text.split("Next motion deltas:")[-1]
    else:
        pred_text = gen_text
        
    next_deltas = parse_one_frame(pred_text)
    
    next_frame_coords = {}
    log_msg = f"Step {step+1:02d}: "
    
    for joint in TARGET_JOINTS:
        if joint in last_frame:
            cur_x, cur_y = last_frame[joint]
            
            if joint in next_deltas:
                dx, dy = next_deltas[joint]
                
                if abs(dx) < MOVEMENT_THRESHOLD: dx = 0.0
                if abs(dy) < MOVEMENT_THRESHOLD: dy = 0.0
                
                dx = max(-VELOCITY_LIMIT, min(VELOCITY_LIMIT, dx))
                dy = max(-VELOCITY_LIMIT, min(VELOCITY_LIMIT, dy))
            else:
                dx, dy = 0.0, 0.0
            
            new_x = cur_x + dx
            new_y = cur_y + dy
            
            if ("ANKLE" in joint or "HEEL" in joint or "FOOT" in joint) and new_y > FLOOR_Y:
                new_y = FLOOR_Y
            
            next_frame_coords[joint] = [new_x, new_y]
            
            if joint == "LEFT_HIP":
                log_msg += f"Hip Δ({dx:.1f},{dy:.1f}) "
    
    print(log_msg)
    current_seq.append(next_frame_coords)
    predicted_frames.append(next_frame_coords)

print("=" * 60)
print("✅ Prediction Loop Complete!")

# 저장 및 시각화 코드 (기존과 동일)
output_data = {
    "original_fps": data.get("fps", 30),
    "predicted_frames": PREDICT_COUNT,
    "pose_sequence": predicted_frames
}
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=2, ensure_ascii=False)
print(f"💾 Results saved to: {output_json_path}")

vis_sequence = [start_frame] + predicted_frames
all_x = [c[0] for f in vis_sequence for c in f.values()]
if all_x:
    fig, ax = plt.subplots(figsize=(6, 6))
    def update(frame_idx):
        ax.clear()
        ax.set_xlim(0, 1000)
        ax.set_ylim(1000, 0)
        ax.axhline(y=1000, color='gray', linestyle='--', linewidth=1, label="Floor")
        ax.grid(True)
        ax.set_title(f"Prediction: {frame_idx}/{len(vis_sequence)}")
        curr = vis_sequence[frame_idx]
        for j, (x, y) in curr.items():
            color = 'red' if 'LEFT' in j else 'blue'
            ax.plot(x, y, "o", ms=5, color=color)
    ani = FuncAnimation(fig, update, frames=len(vis_sequence), interval=100)
    plt.show()