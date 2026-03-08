import argparse
import glob
import json
import os
import urllib.request
from typing import Dict, List, Tuple

RECOMMENDED_MEDIAPIPE = "0.10.32"
DEFAULT_TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Finetuning/Prompt/Test 코드에서 공통으로 쓰는 핵심 관절
REQUIRED_JOINTS = [
    "RIGHT_SHOULDER",
    "RIGHT_ELBOW",
    "RIGHT_WRIST",
    "LEFT_SHOULDER",
    "LEFT_ELBOW",
    "LEFT_WRIST",
]


def _download_task_model(model_url: str, dst_path: str) -> bool:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        urllib.request.urlretrieve(model_url, dst_path)
        return True
    except Exception:
        return False


def _resolve_task_model_path(task_model: str, auto_download: bool) -> str:
    if task_model and os.path.exists(task_model):
        return task_model

    candidates = [
        "./pose_landmarker_lite.task",
        "./pose_landmarker_full.task",
        "./pose_landmarker_heavy.task",
        "./models/pose_landmarker_lite.task",
        "./models/pose_landmarker_full.task",
        "./models/pose_landmarker_heavy.task",
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand

    if auto_download:
        dst = "./models/pose_landmarker_lite.task"
        if _download_task_model(DEFAULT_TASK_MODEL_URL, dst):
            return dst

    return ""


def extract_pose_sequence_from_mp4(
    mp4_path: str,
    min_visibility: float = 0.3,
    required_only: bool = False,
    task_model: str = "",
    auto_download_model: bool = True,
) -> List[Dict[str, List[float]]]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python is required. Install with: pip install opencv-python"
        ) from exc

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required. Install with: pip install mediapipe"
        ) from exc

    mp_ver = getattr(mp, "__version__", "unknown")

    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return _extract_with_solutions(
            mp4_path=mp4_path,
            mp=mp,
            min_visibility=min_visibility,
            required_only=required_only,
        )

    if hasattr(mp, "tasks") and hasattr(mp.tasks, "vision"):
        resolved_model = _resolve_task_model_path(task_model, auto_download_model)
        return _extract_with_tasks_vision(
            mp4_path=mp4_path,
            mp=mp,
            min_visibility=min_visibility,
            required_only=required_only,
            task_model=resolved_model,
        )

    raise RuntimeError(
        "Unsupported mediapipe package layout. "
        "Expected either 'mp.solutions.pose' or 'mp.tasks.vision'. "
        f"Current mediapipe version: {mp_ver}."
    )


def _extract_with_solutions(
    mp4_path: str,
    mp,
    min_visibility: float,
    required_only: bool,
) -> List[Dict[str, List[float]]]:
    import cv2

    landmark_names = [lm.name for lm in mp.solutions.pose.PoseLandmark]
    pose_sequence: List[Dict[str, List[float]]] = []

    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    with mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)
            if not result.pose_landmarks:
                continue

            landmarks = result.pose_landmarks.landmark
            frame_data: Dict[str, List[float]] = {}
            for idx, lm in enumerate(landmarks):
                if lm.visibility < min_visibility:
                    continue
                joint_name = landmark_names[idx]
                if required_only and joint_name not in REQUIRED_JOINTS:
                    continue
                frame_data[joint_name] = [round(float(lm.x), 6), round(float(lm.y), 6)]

            if frame_data:
                pose_sequence.append(frame_data)

    cap.release()
    return pose_sequence


def _extract_with_tasks_vision(
    mp4_path: str,
    mp,
    min_visibility: float,
    required_only: bool,
    task_model: str,
) -> List[Dict[str, List[float]]]:
    import cv2

    if not task_model:
        raise RuntimeError(
            "This mediapipe build uses tasks API and requires a pose landmarker .task model.\n"
            "Pass --task-model <path> or place one of these files in project root/models:\n"
            "  pose_landmarker_lite.task / pose_landmarker_full.task / pose_landmarker_heavy.task\n"
            f"Recommended download URL:\n  {DEFAULT_TASK_MODEL_URL}"
        )
    if not os.path.exists(task_model):
        raise RuntimeError(f"--task-model not found: {task_model}")

    vision = mp.tasks.vision
    base_options = mp.tasks.BaseOptions(model_asset_path=task_model)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    landmark_names = [lm.name for lm in vision.PoseLandmark]

    pose_sequence: List[Dict[str, List[float]]] = []
    cap = cv2.VideoCapture(mp4_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {mp4_path}")

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int((frame_idx / fps) * 1000)
            frame_idx += 1

            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if not result.pose_landmarks:
                continue

            first_pose = result.pose_landmarks[0]
            frame_data: Dict[str, List[float]] = {}
            for idx, lm in enumerate(first_pose):
                visibility = getattr(lm, "visibility", 1.0)
                if visibility < min_visibility:
                    continue
                joint_name = landmark_names[idx]
                if required_only and joint_name not in REQUIRED_JOINTS:
                    continue
                frame_data[joint_name] = [round(float(lm.x), 6), round(float(lm.y), 6)]

            if frame_data:
                pose_sequence.append(frame_data)

    cap.release()
    return pose_sequence


def validate_pose_sequence(
    pose_sequence: List[Dict[str, List[float]]],
    strict: bool = True,
) -> Tuple[bool, List[str]]:
    errors: List[str] = []

    if not isinstance(pose_sequence, list) or not pose_sequence:
        errors.append("pose_sequence must be a non-empty list.")
        return False, errors

    for i, frame in enumerate(pose_sequence):
        if not isinstance(frame, dict):
            errors.append(f"Frame {i}: must be dict.")
            continue
        for joint, xy in frame.items():
            if not isinstance(joint, str):
                errors.append(f"Frame {i}: joint name must be str.")
            if not (isinstance(xy, list) and len(xy) == 2):
                errors.append(f"Frame {i} {joint}: coordinate must be [x, y].")
                continue
            x, y = xy
            if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                errors.append(f"Frame {i} {joint}: x/y must be numeric.")

    # finetuning/test 코드 정합성: 핵심 6개 관절이 충분히 자주 등장해야 함
    joint_counts = {j: 0 for j in REQUIRED_JOINTS}
    for frame in pose_sequence:
        for j in REQUIRED_JOINTS:
            if j in frame:
                joint_counts[j] += 1

    min_frames_for_required = 3 if strict else 1
    for j, c in joint_counts.items():
        if c < min_frames_for_required:
            errors.append(
                f"Required joint '{j}' found in only {c} frame(s), need >= {min_frames_for_required}."
            )

    # 최소 길이: delta 변환 코드가 len(seq) < 3이면 스킵함
    if len(pose_sequence) < 3:
        errors.append("Need at least 3 valid frames for finetuning delta conversion.")

    return len(errors) == 0, errors


def save_pose_json(output_path: str, pose_sequence: List[Dict[str, List[float]]]) -> None:
    payload = {"pose_sequence": pose_sequence}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def process_one_file(
    mp4_path: str,
    output_dir: str,
    min_visibility: float,
    strict: bool,
    required_only: bool,
    task_model: str,
    auto_download_model: bool,
) -> Tuple[bool, str]:
    pose_sequence = extract_pose_sequence_from_mp4(
        mp4_path=mp4_path,
        min_visibility=min_visibility,
        required_only=required_only,
        task_model=task_model,
        auto_download_model=auto_download_model,
    )
    ok, errors = validate_pose_sequence(pose_sequence, strict=strict)
    if not ok:
        return False, f"{os.path.basename(mp4_path)} -> invalid: {'; '.join(errors)}"

    base = os.path.splitext(os.path.basename(mp4_path))[0]
    out_path = os.path.join(output_dir, f"{base}.json")
    save_pose_json(out_path, pose_sequence)
    return True, f"{os.path.basename(mp4_path)} -> {out_path} ({len(pose_sequence)} frames)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert MP4 videos into MediaPipe pose_sequence JSON files."
    )
    parser.add_argument("--input-dir", default="./Custom_Data", help="Directory containing mp4 files")
    parser.add_argument("--output-dir", default="./Custom_Data", help="Directory to save json files")
    parser.add_argument("--pattern", default="*.mp4", help="Glob pattern for input videos")
    parser.add_argument("--min-visible", type=float, default=0.3, help="Minimum landmark visibility")
    parser.add_argument(
        "--required-only",
        action="store_true",
        help="Save only REQUIRED_JOINTS (6 upper-body joints). Default saves all visible joints.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Relax validation rule for required joints.",
    )
    parser.add_argument(
        "--validate-json-only",
        action="store_true",
        help="Skip MP4 extraction and validate existing JSON files in --output-dir.",
    )
    parser.add_argument(
        "--json-pattern",
        default="*.json",
        help="Glob pattern used with --validate-json-only.",
    )
    parser.add_argument(
        "--task-model",
        default="",
        help="Path to pose landmarker .task model (required for mp.tasks-only mediapipe builds).",
    )
    parser.add_argument(
        "--no-auto-download-model",
        action="store_true",
        help="Disable automatic download of pose_landmarker_lite.task when no local model is found.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search subdirectories under --input-dir. Mirrors folder structure in --output-dir.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    try:
        import mediapipe as mp  # local check for user guidance
        mp_ver = getattr(mp, "__version__", "unknown")
        if mp_ver != RECOMMENDED_MEDIAPIPE:
            print(
                f"[WARN] mediapipe=={mp_ver} detected. "
                f"Recommended for this project: mediapipe=={RECOMMENDED_MEDIAPIPE}."
            )
    except Exception:
        pass

    if args.validate_json_only:
        json_paths = sorted(glob.glob(os.path.join(args.output_dir, args.json_pattern)))
        if not json_paths:
            raise SystemExit(
                f"No JSON files found: {os.path.join(args.output_dir, args.json_pattern)}"
            )

        strict = not args.no_strict
        ok_count = 0
        fail_count = 0
        for path in json_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                pose_sequence = payload.get("pose_sequence", [])
            except Exception as exc:
                print(f"[FAIL] {os.path.basename(path)} -> read error: {exc}")
                fail_count += 1
                continue

            ok, errors = validate_pose_sequence(pose_sequence, strict=strict)
            if ok:
                print(f"[OK] {os.path.basename(path)}")
                ok_count += 1
            else:
                print(f"[FAIL] {os.path.basename(path)} -> {'; '.join(errors)}")
                fail_count += 1

        print(f"\nValidation done. success={ok_count}, failed={fail_count}")
        if fail_count > 0:
            raise SystemExit(1)
        return

    strict = not args.no_strict
    auto_download_model = not args.no_auto_download_model
    success = 0
    failed = 0

    if args.recursive:
        # os.walk로 하위 폴더를 모두 순회하며 MP4 수집
        import fnmatch
        tasks = []
        for dirpath, _, filenames in os.walk(args.input_dir):
            for filename in sorted(filenames):
                if fnmatch.fnmatch(filename.lower(), args.pattern.lower()):
                    mp4_path = os.path.join(dirpath, filename)
                    # input_dir 기준 상대 경로로 output 폴더 미러링
                    rel_dir = os.path.relpath(dirpath, args.input_dir)
                    out_dir = os.path.join(args.output_dir, rel_dir) if rel_dir != "." else args.output_dir
                    tasks.append((mp4_path, out_dir))

        if not tasks:
            raise SystemExit(f"No videos found recursively under: {args.input_dir}")

        print(f"[INFO] Found {len(tasks)} video(s) across subdirectories.")
        for mp4_path, out_dir in tasks:
            os.makedirs(out_dir, exist_ok=True)
            ok, msg = process_one_file(
                mp4_path=mp4_path,
                output_dir=out_dir,
                min_visibility=args.min_visible,
                strict=strict,
                required_only=args.required_only,
                task_model=args.task_model,
                auto_download_model=auto_download_model,
            )
            if ok:
                success += 1
                print(f"[OK] {msg}")
            else:
                failed += 1
                print(f"[FAIL] {msg}")
    else:
        paths = sorted(glob.glob(os.path.join(args.input_dir, args.pattern)))
        if not paths:
            raise SystemExit(f"No videos found: {os.path.join(args.input_dir, args.pattern)}")

        for mp4_path in paths:
            ok, msg = process_one_file(
                mp4_path=mp4_path,
                output_dir=args.output_dir,
                min_visibility=args.min_visible,
                strict=strict,
                required_only=args.required_only,
                task_model=args.task_model,
                auto_download_model=auto_download_model,
            )
            if ok:
                success += 1
                print(f"[OK] {msg}")
            else:
                failed += 1
                print(f"[FAIL] {msg}")

    print(f"\nDone. success={success}, failed={failed}")
    if failed > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
