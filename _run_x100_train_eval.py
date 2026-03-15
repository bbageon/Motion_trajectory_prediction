"""
×100 스케일 실험 순차 실행 스크립트.
1. Delta ×100 데이터 생성
2. Absolute ×100 데이터 생성
3. Delta ×100 학습
4. Absolute ×100 학습
5. Delta ×100 평가
6. Absolute ×100 평가

각 단계의 로그를 logs/ 폴더에 저장.
실행: conda run -n motionqa python _run_x100_train_eval.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASELINE = ROOT / "BaseLine"
DELTA_DIR = ROOT / "delta"
ABSOLUTE_DIR = ROOT / "absolute"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)


def run_step(label: str, args: list, cwd: Path) -> bool:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"{timestamp}_{label.replace(' ', '_')}.log"

    print(f"\n{'='*60}")
    print(f"[RUN] {label}")
    print(f"[CWD] {cwd}")
    print(f"[LOG] {log_file}")
    print(f"{'='*60}", flush=True)

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"# {label}\n")
        lf.write(f"# started: {datetime.now()}\n")
        lf.write(f"# cwd: {cwd}\n")
        lf.write(f"# cmd: {' '.join(str(a) for a in args)}\n\n")
        lf.flush()

        process = subprocess.Popen(
            args,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8:replace"},
        )

        for line in process.stdout:
            print(line, end="", flush=True)
            lf.write(line)
            lf.flush()

        process.wait()

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n# finished: {datetime.now()}\n")
        lf.write(f"# return code: {process.returncode}\n")

    if process.returncode != 0:
        print(f"\n[ERROR] '{label}' failed (code {process.returncode}). Log: {log_file}")
        return False

    print(f"\n[DONE] '{label}' completed. Log: {log_file}")
    return True


STEPS = [
    {
        "label": "01_generate_delta_x100",
        "cwd": BASELINE,
        "args": [sys.executable, "05_delta_convertToFinetuning_x100_fileLevelSplit.py"],
    },
    {
        "label": "02_generate_absolute_x100",
        "cwd": BASELINE,
        "args": [sys.executable, "05_abs_convertToFinetuning_x100_fileLevelSplit.py"],
    },
    {
        "label": "03_train_delta_x100",
        "cwd": BASELINE,
        "args": [
            sys.executable, "06_finetune_llama2_lora.py",
            "--train-jsonl", "x100 데이터/finetune_dataset_delta_x100_fileLevel_train.jsonl",
            "--val-jsonl",   "x100 데이터/finetune_dataset_delta_x100_fileLevel_val.jsonl",
            "--output-dir",  "x100_delta_fileLevel_lora",
            "--max-length",  "4096",
            "--learning-rate", "5e-5",
            "--epochs",      "6",
            "--batch-size",  "1",
            "--grad-accum",  "8",
            "--lora-r",      "16",
            "--lora-alpha",  "32",
            "--lora-dropout", "0.05",
            "--token-io-mode", "lm_head_special_only",
        ],
    },
    {
        "label": "04_train_absolute_x100",
        "cwd": BASELINE,
        "args": [
            sys.executable, "06_finetune_llama2_lora.py",
            "--train-jsonl", "x100 데이터/finetune_dataset_absolute_x100_fileLevel_train.jsonl",
            "--val-jsonl",   "x100 데이터/finetune_dataset_absolute_x100_fileLevel_val.jsonl",
            "--output-dir",  "x100_absolute_fileLevel_lora",
            "--max-length",  "4096",
            "--learning-rate", "5e-5",
            "--epochs",      "6",
            "--batch-size",  "1",
            "--grad-accum",  "8",
            "--lora-r",      "16",
            "--lora-alpha",  "32",
            "--lora-dropout", "0.05",
            "--token-io-mode", "lm_head_special_only",
        ],
    },
    {
        "label": "05_eval_delta_x100",
        "cwd": DELTA_DIR,
        "args": [
            sys.executable, "01_test2_delta_x100_evalation.py",
            "--append-target-prefix",
            "--accumulate-to-absolute",
            "--output-report", "./delta_x100_eval_report.json",
        ],
    },
    {
        "label": "06_eval_absolute_x100",
        "cwd": ABSOLUTE_DIR,
        "args": [
            sys.executable, "01_test2_absolute_x100_evalation.py",
            "--append-target-prefix",
            "--output-report", "./absolute_x100_eval_report.json",
        ],
    },
]


def main():
    print(f"[START] ×100 스케일 전체 파이프라인 시작: {datetime.now()}")
    print(f"[LOG DIR] {LOG_DIR}")

    results = {}
    for step in STEPS:
        ok = run_step(step["label"], step["args"], step["cwd"])
        results[step["label"]] = "OK" if ok else "FAILED"
        if not ok:
            print(f"[WARN] '{step['label']}' 실패 - 다음 단계로 계속 진행합니다.")

    print(f"\n{'='*60}")
    print(f"[ALL DONE] 파이프라인 완료: {datetime.now()}")
    print(f"[LOGS] {LOG_DIR}")
    print("\n[결과 요약]")
    for label, status in results.items():
        mark = "OK" if status == "OK" else "FAIL"
        print(f"  [{mark}] {label}: {status}")
    print(f"\n  - delta eval report:    {DELTA_DIR / 'delta_x100_eval_report.json'}")
    print(f"  - absolute eval report: {ABSOLUTE_DIR / 'absolute_x100_eval_report.json'}")


if __name__ == "__main__":
    main()
