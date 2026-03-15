"""
Delta ×100 모델 전용 평가 스크립트.
- 학습 시 delta × 100으로 토큰화된 모델 전용
- 디코딩 후 ÷100 역변환 → 원본 좌표 공간(0~1)에서 MAE 계산
- noScale ×100 데이터 / x100 LoRA 모델 기본 경로
"""

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MARKER = "Next motion deltas:"
DEC_DIGITS = 5
DEC_SCALE = 10**DEC_DIGITS
X100_FACTOR = 100.0  # 학습 시 곱한 값 → 평가 시 나눔


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delta x100 model evaluation on test JSONL.")
    parser.add_argument("--base-model-dir", default="../Meta-Llama-3.1-8B_tokenizerExtension")
    parser.add_argument("--lora-model-dir", default="../BaseLine/x100_delta_fileLevel_lora")
    parser.add_argument(
        "--test-jsonl",
        default="../BaseLine/x100 데이터/finetune_dataset_delta_x100_fileLevel_test.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--min-new-tokens", type=int, default=32)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all samples.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling generation.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--output-report", default="./delta_x100_eval_report.json")
    parser.add_argument(
        "--failure-log-jsonl",
        default="",
        help="Optional JSONL path to store per-sample failure records. Defaults to <output-report>_failures.jsonl",
    )
    parser.add_argument(
        "--append-target-prefix",
        action="store_true",
        help="Append '\\nNext motion deltas:' to each prompt before generation.",
    )
    parser.add_argument(
        "--debug-failures",
        type=int,
        default=3,
        help="Print raw model outputs for first N parse failures.",
    )
    parser.add_argument(
        "--accumulate-to-absolute",
        action="store_true",
        help=(
            "Accumulate predicted deltas from the last observed frame to get absolute "
            "coordinates, then compute MAE in absolute space."
        ),
    )
    parser.add_argument(
        "--obs-end-index",
        type=int,
        default=10,
        help="Frame index (0-based) of the last observed frame used as accumulation baseline.",
    )
    return parser.parse_args()


def decode_token_number(token_str: str) -> float:
    """토큰 → 실수 변환 후 ÷100 역변환."""
    pattern = r"(-?)\[NUM\]\[INT\](\d{3})\[SEP\]\[DEC\](\d{5})\[ENDNUM\]"
    m = re.search(pattern, token_str)
    if not m:
        raise ValueError(f"Invalid token format: {token_str}")
    sign = -1 if m.group(1) == "-" else 1
    raw = sign * (int(m.group(2)) + int(m.group(3)) / DEC_SCALE)
    return round(raw / X100_FACTOR, DEC_DIGITS + 1)


def parse_deltas(text: str) -> dict[str, list[tuple[float, float]]]:
    start = text.find(MARKER)
    if start != -1:
        text = text[start + len(MARKER):]
    text = text.strip()
    if not text:
        return {}
    frame_pat = re.compile(
        r"\(\s*(-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{5}\[ENDNUM\])\s*"
        r",\s*(-?\[NUM\]\[INT\]\d{3}\[SEP\]\[DEC\]\d{5}\[ENDNUM\])\s*\)"
    )
    joint_pat = re.compile(r"([A-Z][A-Z_]*):")
    result: dict[str, list[tuple[float, float]]] = {}
    matches = list(joint_pat.finditer(text))
    for idx, m in enumerate(matches):
        joint = m.group(1)
        seg_start = m.end()
        seg_end = matches[idx + 1].start() if (idx + 1) < len(matches) else len(text)
        seg = text[seg_start:seg_end]
        frames = []
        for x_tok, y_tok in frame_pat.findall(seg):
            frames.append((decode_token_number(x_tok), decode_token_number(y_tok)))
        if frames:
            result[joint] = frames
    return result


def safe_console_text(text: str) -> str:
    """Convert text to current console encoding safely to avoid print-time crashes."""
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    return text.encode(enc, errors="replace").decode(enc, errors="replace")


def classify_format_failure(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return "empty_output"
    if "[NUM]" not in t:
        return "no_num_token"
    if "[SEP][DEC]" not in t and "[SEP]" in t:
        return "missing_dec_after_sep"
    if "[ENDNUM]" not in t:
        return "missing_endnum"
    if "#" in t or "\ufffd" in t:
        return "garbage_symbols"
    return "unparsable_format"


def load_baseline_frame(source_file: str, obs_end_index: int) -> dict:
    with open(source_file, "r", encoding="utf-8") as f:
        raw = json.load(f)
    pose_seq = raw["pose_sequence"]
    idx = min(obs_end_index, len(pose_seq) - 1)
    return {k: tuple(v) for k, v in pose_seq[idx].items()}


def delta_to_absolute(
    baseline_frame: dict,
    deltas: dict[str, list[tuple[float, float]]],
) -> dict[str, list[tuple[float, float]]]:
    abs_seq: dict[str, list[tuple[float, float]]] = {}
    for joint, seq in deltas.items():
        if joint not in baseline_frame:
            continue
        bx, by = baseline_frame[joint]
        x, y = float(bx), float(by)
        coords = []
        for dx, dy in seq:
            x = round(x + dx, 6)
            y = round(y + dy, 6)
            coords.append((x, y))
        abs_seq[joint] = coords
    return abs_seq


def evaluate_one(
    pred: dict[str, list[tuple[float, float]]],
    gt: dict[str, list[tuple[float, float]]],
) -> dict:
    err_x, err_y, dist = [], [], []
    horizon_dist = defaultdict(list)
    common_joints = sorted(set(pred.keys()) & set(gt.keys()))
    aligned_points = 0
    for joint in common_joints:
        n = min(len(pred[joint]), len(gt[joint]))
        for t in range(n):
            x_p, y_p = pred[joint][t]
            x_g, y_g = gt[joint][t]
            err_x.append(abs(x_p - x_g))
            err_y.append(abs(y_p - y_g))
            d = math.sqrt((x_p - x_g) ** 2 + (y_p - y_g) ** 2)
            dist.append(d)
            horizon_dist[t + 1].append(d)
            aligned_points += 1
    return {
        "common_joints": len(common_joints),
        "aligned_points": aligned_points,
        "mae_x": sum(err_x) / len(err_x) if err_x else None,
        "mae_y": sum(err_y) / len(err_y) if err_y else None,
        "mae_xy": (sum(err_x) + sum(err_y)) / (len(err_x) + len(err_y)) if err_x else None,
        "mean_dist": sum(dist) / len(dist) if dist else None,
        "horizon_dist": {k: (sum(v) / len(v) if v else None) for k, v in horizon_dist.items()},
        "raw_err_x": err_x,
        "raw_err_y": err_y,
        "raw_dist": dist,
        "raw_horizon_dist": horizon_dist,
    }


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    base_model_dir = (script_dir / args.base_model_dir).resolve()
    lora_model_dir = (script_dir / args.lora_model_dir).resolve()
    test_jsonl = (script_dir / args.test_jsonl).resolve()
    output_report = (script_dir / args.output_report).resolve()
    failure_log_jsonl = (
        (script_dir / args.failure_log_jsonl).resolve()
        if args.failure_log_jsonl
        else output_report.with_name(f"{output_report.stem}_failures.jsonl")
    )
    accumulate_to_absolute = args.accumulate_to_absolute
    obs_end_index = args.obs_end_index

    if not base_model_dir.exists():
        raise FileNotFoundError(f"base model dir not found: {base_model_dir}")
    if not lora_model_dir.exists():
        raise FileNotFoundError(f"lora model dir not found: {lora_model_dir}")
    if not test_jsonl.exists():
        raise FileNotFoundError(f"test jsonl not found: {test_jsonl}")

    print(f"[INFO] base model: {base_model_dir}")
    print(f"[INFO] lora model: {lora_model_dir}")
    print(f"[INFO] test jsonl: {test_jsonl}")
    print(f"[INFO] x100 inverse scaling: ENABLED (decoded value ÷ {X100_FACTOR})")
    print(f"[INFO] MAE computed in original coordinate space (0~1)")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), local_files_only=True, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, str(lora_model_dir))
    model.eval()
    model.to(device)
    print(f"[INFO] model ready on {device}")

    with open(test_jsonl, "r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    if not samples:
        raise ValueError("No samples to evaluate.")

    total = len(samples)
    parse_ok = 0
    aligned_ok = 0
    failure_printed = 0
    all_err_x, all_err_y, all_dist = [], [], []
    all_horizon = defaultdict(list)
    action_stats = defaultdict(lambda: {"total": 0, "parse_ok": 0, "aligned_ok": 0, "dist": []})
    failure_reason_counts = defaultdict(int)
    failure_records = []

    for idx, row in enumerate(samples, 1):
        prompt = row.get("prompt", "")
        if args.append_target_prefix:
            prompt = f"{prompt}\n{MARKER}"
        gt_text = row.get("completion", "")
        action = row.get("action", "unknown")
        action_stats[action]["total"] += 1

        mode_label = "abs" if accumulate_to_absolute else "delta"
        parse_label = "fail(format_collapse)"
        aligned_points = 0
        pred_text = ""
        pred_deltas = {}
        source_file = row.get("source_file", "")

        try:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            input_len = inputs["input_ids"].shape[1]

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "min_new_tokens": args.min_new_tokens,
                "repetition_penalty": args.repetition_penalty,
                "pad_token_id": tokenizer.eos_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": False,
            }
            if args.do_sample:
                gen_kwargs.update({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p})

            with torch.no_grad():
                outputs = model.generate(**inputs, **gen_kwargs)

            pred_text = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=False)
            pred_deltas = parse_deltas(pred_text)
            gt_deltas = parse_deltas(gt_text)

            if pred_deltas:
                parse_label = "ok"
                parse_ok += 1
                action_stats[action]["parse_ok"] += 1
            else:
                fail_reason = classify_format_failure(pred_text)
                failure_reason_counts[fail_reason] += 1
                failure_records.append(
                    {
                        "sample_index": idx,
                        "action": action,
                        "source_file": source_file,
                        "mode": mode_label,
                        "failure_type": "format_collapse",
                        "failure_reason": fail_reason,
                        "prompt_head": prompt[:200].replace("\n", " "),
                        "raw_output_head": pred_text[:800].replace("\n", "\\n"),
                    }
                )
                if failure_printed < args.debug_failures:
                    failure_printed += 1
                    print(f"\n[DEBUG parse_fail #{failure_printed}] action={action}")
                    print("[DEBUG prompt head]", safe_console_text(prompt[:200].replace("\n", " ")))
                    print("[DEBUG raw output head]", safe_console_text(pred_text[:800].replace("\n", "\\n")))

            if accumulate_to_absolute:
                baseline_frame = {}
                try:
                    baseline_frame = load_baseline_frame(source_file, obs_end_index)
                except Exception as e:
                    print(f"  [ABS] WARNING: could not load baseline ({e})")
                eval_pred = delta_to_absolute(baseline_frame, pred_deltas) if baseline_frame else {}
                eval_gt = delta_to_absolute(baseline_frame, gt_deltas) if baseline_frame else {}
            else:
                eval_pred, eval_gt = pred_deltas, gt_deltas

            m = evaluate_one(eval_pred, eval_gt)
            aligned_points = m["aligned_points"]
            if aligned_points > 0:
                aligned_ok += 1
                action_stats[action]["aligned_ok"] += 1
                all_err_x.extend(m["raw_err_x"])
                all_err_y.extend(m["raw_err_y"])
                all_dist.extend(m["raw_dist"])
                action_stats[action]["dist"].extend(m["raw_dist"])
                for h, values in m["raw_horizon_dist"].items():
                    all_horizon[h].extend(values)
        except Exception as e:
            parse_label = "error(runtime)"
            fail_reason = f"runtime_error:{type(e).__name__}"
            failure_reason_counts[fail_reason] += 1
            failure_records.append(
                {
                    "sample_index": idx,
                    "action": action,
                    "source_file": source_file,
                    "mode": mode_label,
                    "failure_type": "runtime_error",
                    "failure_reason": fail_reason,
                    "error_message": str(e),
                    "prompt_head": prompt[:200].replace("\n", " "),
                    "raw_output_head": pred_text[:800].replace("\n", "\\n"),
                }
            )
            print(f"[WARN] sample {idx}/{total} runtime error: {safe_console_text(str(e))}")

        print(f"[{idx}/{total}] action={action} mode={mode_label} parse={parse_label} aligned_points={aligned_points}")

    horizon_summary = {
        str(h): (sum(v) / len(v) if v else None)
        for h, v in sorted(all_horizon.items(), key=lambda x: x[0])
    }
    action_summary = {}
    for action, s in sorted(action_stats.items()):
        action_summary[action] = {
            "total": s["total"],
            "parse_success_rate": (s["parse_ok"] / s["total"]) if s["total"] else 0.0,
            "aligned_success_rate": (s["aligned_ok"] / s["total"]) if s["total"] else 0.0,
            "mean_dist": (sum(s["dist"]) / len(s["dist"])) if s["dist"] else None,
        }

    report = {
        "num_samples": total,
        "parse_success_rate": parse_ok / total,
        "aligned_success_rate": aligned_ok / total,
        "overall_mae_x": (sum(all_err_x) / len(all_err_x)) if all_err_x else None,
        "overall_mae_y": (sum(all_err_y) / len(all_err_y)) if all_err_y else None,
        "overall_mae_xy": ((sum(all_err_x) + sum(all_err_y)) / (len(all_err_x) + len(all_err_y))) if all_err_x else None,
        "overall_mean_dist": (sum(all_dist) / len(all_dist)) if all_dist else None,
        "horizon_mean_dist": horizon_summary,
        "action_summary": action_summary,
        "num_failures": len(failure_records),
        "failure_reason_counts": dict(sorted(failure_reason_counts.items(), key=lambda x: x[0])),
        "failure_samples": failure_records,
        "config": {
            "base_model_dir": str(base_model_dir),
            "lora_model_dir": str(lora_model_dir),
            "test_jsonl": str(test_jsonl),
            "failure_log_jsonl": str(failure_log_jsonl),
            "x100_inverse_scaling": True,
            "x100_factor": X100_FACTOR,
            "accumulate_to_absolute": accumulate_to_absolute,
            "obs_end_index": obs_end_index,
            "repetition_penalty": args.repetition_penalty,
            "device": device,
        },
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    with open(failure_log_jsonl, "w", encoding="utf-8") as f:
        for row in failure_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("\n[SUMMARY]")
    print(f"  samples: {report['num_samples']}")
    print(f"  parse success: {report['parse_success_rate']:.2%}")
    print(f"  aligned success: {report['aligned_success_rate']:.2%}")
    print(f"  MAE(x): {report['overall_mae_x']}")
    print(f"  MAE(y): {report['overall_mae_y']}")
    print(f"  MAE(xy): {report['overall_mae_xy']}")
    print(f"  Mean distance: {report['overall_mean_dist']}")
    print(f"  failures: {report['num_failures']}")
    if accumulate_to_absolute:
        print(f"  [NOTE] Metrics in ABSOLUTE space (delta accumulated from frame {obs_end_index}, ÷{X100_FACTOR} applied)")
    else:
        print(f"  [NOTE] Metrics in DELTA space (÷{X100_FACTOR} applied)")
    print(f"  report saved: {output_report}")
    print(f"  failure log saved: {failure_log_jsonl}")


if __name__ == "__main__":
    main()
