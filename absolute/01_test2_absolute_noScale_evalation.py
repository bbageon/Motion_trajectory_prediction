"""
Absolute noScale 모델 전용 평가 스크립트.
- 역 스케일링 하드코딩 OFF (noScale 학습 모델과 쌍)
- 기본 경로: noScale 데이터 / noScale LoRA 모델
- 기존 01_test2_absolute_evalation.py 와 동일 구조, scaling 고정 비활성화
"""

import argparse
import json
import math
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


MARKER = "Next absolute coordinates:"
DEC_DIGITS = 5
DEC_SCALE = 10**DEC_DIGITS


def _safe_debug_text(text: str, limit: int) -> str:
    snippet = text[:limit].replace("\n", "\\n")
    out_enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        return snippet.encode(out_enc, errors="backslashreplace").decode(out_enc, errors="replace")
    except LookupError:
        return snippet.encode("utf-8", errors="backslashreplace").decode("utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Absolute noScale model evaluation on test JSONL.")
    parser.add_argument("--base-model-dir", default="../Meta-Llama-3.1-8B_tokenizerExtension")
    parser.add_argument("--lora-model-dir", default="../BaseLine/noScale_absolute_fileLevel_lora")
    parser.add_argument(
        "--test-jsonl",
        default="../BaseLine/스케일링 미적용 데이터/finetune_dataset_absolute_noScale_fileLevel_test.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--min-new-tokens", type=int, default=1024)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means evaluate all samples.")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling generation.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--output-report", default="./absolute_noScale_eval_report.json")
    parser.add_argument(
        "--append-target-prefix",
        action="store_true",
        help="Append '\\nNext absolute coordinates:' to each prompt before generation.",
    )
    parser.add_argument(
        "--debug-failures",
        type=int,
        default=3,
        help="Print raw model outputs for first N parse failures.",
    )
    return parser.parse_args()


def decode_token_number(token_str: str) -> float:
    pattern = r"(-?)\[NUM\]\[INT\](\d{3})\[SEP\]\[DEC\](\d{5})\[ENDNUM\]"
    m = re.search(pattern, token_str)
    if not m:
        raise ValueError(f"Invalid token format: {token_str}")
    sign = -1 if m.group(1) == "-" else 1
    return round(sign * (int(m.group(2)) + int(m.group(3)) / DEC_SCALE), DEC_DIGITS + 1)


def _to_ascii_digits(text: str) -> str:
    out = []
    for ch in text:
        if "0" <= ch <= "9":
            out.append(ch)
            continue
        try:
            out.append(str(unicodedata.digit(ch)))
        except (TypeError, ValueError):
            continue
    return "".join(out)


def _parse_numeric_fragment(fragment: str) -> float | None:
    frag = fragment.strip()
    if not frag:
        return None
    m = re.search(
        r"(-?)\s*\[NUM\]\s*\[INT\]\s*(\d{3})\s*\[SEP\]\s*\[DEC\]\s*(\d{5})\s*\[ENDNUM\]", frag
    )
    if m:
        sign = -1 if m.group(1) == "-" else 1
        return round(sign * (int(m.group(2)) + int(m.group(3)) / DEC_SCALE), DEC_DIGITS + 1)
    sign = -1 if re.search(r"[-−﹣－]", frag) else 1
    digits = _to_ascii_digits(frag)
    if len(digits) >= 8:
        core = digits[-8:]
        return round(sign * (int(core[:3]) + int(core[3:8]) / DEC_SCALE), DEC_DIGITS + 1)
    m = re.search(r"[-+]?\d+(?:\.\d+)?", frag)
    if m:
        try:
            return round(float(m.group(0)), DEC_DIGITS + 1)
        except ValueError:
            return None
    return None


def parse_coords(text: str) -> dict[str, list[tuple[float, float]]]:
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
        frames: list[tuple[float, float]] = []
        for x_tok, y_tok in frame_pat.findall(seg):
            frames.append((decode_token_number(x_tok), decode_token_number(y_tok)))
        if not frames:
            for inner in re.findall(r"\(([^()]*)\)", seg):
                parts = re.split(r"[,،|]", inner, maxsplit=1)
                if len(parts) < 2:
                    continue
                x = _parse_numeric_fragment(parts[0])
                y = _parse_numeric_fragment(parts[1])
                if x is None or y is None:
                    continue
                frames.append((x, y))
        if frames:
            result[joint] = frames
    return result


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

    # noScale 전용: 역 스케일링 비활성화 (하드코딩)
    apply_absolute_scaling = False

    if not base_model_dir.exists():
        raise FileNotFoundError(f"base model dir not found: {base_model_dir}")
    if not lora_model_dir.exists():
        raise FileNotFoundError(f"lora model dir not found: {lora_model_dir}")
    if not test_jsonl.exists():
        raise FileNotFoundError(f"test jsonl not found: {test_jsonl}")

    print(f"[INFO] base model: {base_model_dir}")
    print(f"[INFO] lora model: {lora_model_dir}")
    print(f"[INFO] test jsonl: {test_jsonl}")
    print(f"[INFO] absolute scaling: DISABLED (noScale model)")

    tokenizer = AutoTokenizer.from_pretrained(str(base_model_dir), local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(str(base_model_dir), local_files_only=True, torch_dtype=dtype)
    model = PeftModel.from_pretrained(model, str(lora_model_dir))
    model.eval()
    model.to(device)
    if getattr(model, "generation_config", None) is not None and not args.do_sample:
        model.generation_config.do_sample = False
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
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

    for idx, row in enumerate(samples, 1):
        prompt = row.get("prompt", "")
        if args.append_target_prefix:
            prompt = f"{prompt}\n{MARKER}"
        gt_text = row.get("completion", "")
        action = row.get("action", "unknown")
        action_stats[action]["total"] += 1

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
        pred_coords = parse_coords(pred_text)
        gt_coords = parse_coords(gt_text)
        # noScale: 역변환 없이 raw 값 그대로 사용

        if pred_coords:
            parse_ok += 1
            action_stats[action]["parse_ok"] += 1
        elif failure_printed < args.debug_failures:
            failure_printed += 1
            print(f"\n[DEBUG parse_fail #{failure_printed}] action={action}")
            print("[DEBUG prompt head]", _safe_debug_text(prompt.replace("\n", " "), 200))
            print("[DEBUG raw output head]", _safe_debug_text(pred_text, 800))

        m = evaluate_one(pred_coords, gt_coords)
        if m["aligned_points"] > 0:
            aligned_ok += 1
            action_stats[action]["aligned_ok"] += 1
            all_err_x.extend(m["raw_err_x"])
            all_err_y.extend(m["raw_err_y"])
            all_dist.extend(m["raw_dist"])
            action_stats[action]["dist"].extend(m["raw_dist"])
            for h, values in m["raw_horizon_dist"].items():
                all_horizon[h].extend(values)

        print(f"[{idx}/{total}] action={action} parse={'ok' if pred_coords else 'fail'} aligned_points={m['aligned_points']}")

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
        "config": {
            "base_model_dir": str(base_model_dir),
            "lora_model_dir": str(lora_model_dir),
            "test_jsonl": str(test_jsonl),
            "absolute_scaling_enabled": False,
            "repetition_penalty": args.repetition_penalty,
            "device": device,
        },
    }

    output_report.parent.mkdir(parents=True, exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("\n[SUMMARY]")
    print(f"  samples: {report['num_samples']}")
    print(f"  parse success: {report['parse_success_rate']:.2%}")
    print(f"  aligned success: {report['aligned_success_rate']:.2%}")
    print(f"  MAE(x): {report['overall_mae_x']}")
    print(f"  MAE(y): {report['overall_mae_y']}")
    print(f"  MAE(xy): {report['overall_mae_xy']}")
    print(f"  Mean distance: {report['overall_mean_dist']}")
    print(f"  [NOTE] Metrics in RAW coordinate space (no inverse scaling)")
    print(f"  report saved: {output_report}")


if __name__ == "__main__":
    main()
