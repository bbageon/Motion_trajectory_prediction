"""
선택한 명령만 logs/에 저장하는 실행기.

사용 예시:
  python tools/run_logged.py --label train_absolute_resume --cwd BaseLine -- ^
    C:\\Users\\DSU\\miniconda3\\envs\\motionqa\\python.exe 06_finetune_llama2_lora.py --help
"""

import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command and stream output to both terminal and logs/<timestamp>_<label>.log"
    )
    parser.add_argument("--label", required=True, help="Log label, e.g. train_absolute_x100")
    parser.add_argument("--cwd", default=".", help="Working directory for the command")
    parser.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run. Put '--' before command.",
    )
    args = parser.parse_args()
    if args.cmd and args.cmd[0] == "--":
        args.cmd = args.cmd[1:]
    if not args.cmd:
        parser.error("No command provided. Use: -- <command ...>")
    return args


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    log_dir = root / "logs"
    log_dir.mkdir(exist_ok=True)

    label = args.label.replace(" ", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"{timestamp}_{label}.log"

    workdir = Path(args.cwd)
    if not workdir.is_absolute():
        workdir = (root / workdir).resolve()

    print(f"[RUN] {label}")
    print(f"[CWD] {workdir}")
    print(f"[LOG] {log_file}")

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"# {label}\n")
        lf.write(f"# started: {datetime.now()}\n")
        lf.write(f"# cwd: {workdir}\n")
        lf.write(f"# cmd: {' '.join(args.cmd)}\n\n")
        lf.flush()

        process = subprocess.Popen(
            args.cmd,
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8:replace"},
        )

        assert process.stdout is not None
        out_encoding = sys.stdout.encoding or "utf-8"
        for line in process.stdout:
            try:
                print(line, end="", flush=True)
            except UnicodeEncodeError:
                safe = line.encode(out_encoding, errors="replace").decode(out_encoding, errors="replace")
                print(safe, end="", flush=True)
            lf.write(line)
            lf.flush()

        return_code = process.wait()

    with open(log_file, "a", encoding="utf-8") as lf:
        lf.write(f"\n# finished: {datetime.now()}\n")
        lf.write(f"# return code: {return_code}\n")

    if return_code == 0:
        print(f"[DONE] {label} (code=0)")
    else:
        print(f"[FAIL] {label} (code={return_code})")
    print(f"[LOG] {log_file}")
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
