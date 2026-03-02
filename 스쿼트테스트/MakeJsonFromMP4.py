import cv2
import mediapipe as mp
import json
import os
import glob
from tqdm import tqdm

# -----------------------------
# 1️⃣ 설정 (경로 및 파라미터)
# -----------------------------
INPUT_DIR = "./Custom_Data"
OUTPUT_DIR = "./collected_motions"
TARGET_FPS = 10 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Pose 설정
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6,
    model_complexity=1
)

def process_video(video_path):
    filename = os.path.basename(video_path)
    file_stem = os.path.splitext(filename)[0]
    json_path = os.path.join(OUTPUT_DIR, f"{file_stem}.json")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 파일을 열 수 없습니다: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / original_fps if original_fps > 0 else 0
    
    if original_fps > TARGET_FPS:
        frame_interval = int(round(original_fps / TARGET_FPS))
    else:
        frame_interval = 1 

    pose_data = []
    last_valid_joints = None  # ⭐ 감지 실패 시 사용할 마지막 성공 데이터 저장소

    # -----------------------------
    # 2️⃣ 프레임 순회 및 보간(Interpolation) 추출
    # -----------------------------
    for frame_idx in tqdm(range(total_frames), desc=filename, unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(frame_rgb)

        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            joints = {
                joint.name: [landmarks[joint.value].x, landmarks[joint.value].y]
                for joint in mp_pose.PoseLandmark
            }
            last_valid_joints = joints  # 성공한 데이터 업데이트
            pose_data.append(joints)
        else:
            # ⭐ 핵심: 감지에 실패했다면 직전 프레임의 데이터를 복사하여 삽입
            if last_valid_joints is not None:
                pose_data.append(last_valid_joints)
            else:
                # 영상 시작부터 감지가 안 되는 경우만 빈 딕셔너리 유지
                pose_data.append({})

    cap.release()

    if not pose_data:
        return

    # -----------------------------
    # 3️⃣ JSON 저장
    # -----------------------------
    output_data = {
        "original_fps": round(original_fps, 2),
        "target_fps": TARGET_FPS,
        "saved_fps": round(original_fps / frame_interval, 2),
        "duration_sec": round(duration, 2),
        "total_original_frames": total_frames,
        "num_saved_frames": len(pose_data),
        "frame_interval": frame_interval,
        "pose_sequence": pose_data, 
    }

    with open(json_path, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ 저장 완료: {json_path}")

if __name__ == "__main__":
    video_files = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    for video_path in video_files:
        process_video(video_path)
    cv2.destroyAllWindows()