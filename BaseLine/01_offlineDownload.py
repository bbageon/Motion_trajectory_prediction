# python BaseLine\01_offlineDownload.py --model-preset llama2-7b
# 만약 7b 모델 다운 시, 해당 명령어 사용

import argparse
import os
from pathlib import Path

DEFAULT_REPO_ID = "meta-llama/Meta-Llama-3.1-8B"

PRESET_REPOS = {
    "llama2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama3.1-8b-base": "meta-llama/Meta-Llama-3.1-8B",
    "llama3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}


def model_dir_name(repo_id: str) -> str:
    # e.g. "meta-llama/Llama-2-13b-chat-hf" -> "Llama-2-13b-chat-hf"
    return repo_id.split("/")[-1]


def resolve_local_dir(local_dir_arg: str, repo_id: str) -> str:
    if local_dir_arg.strip():
        return local_dir_arg
    script_dir = Path(__file__).resolve().parent
    target = (script_dir / ".." / model_dir_name(repo_id)).resolve()
    return str(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Llama model snapshot from Hugging Face.")
    parser.add_argument(
        "--model-preset",
        choices=list(PRESET_REPOS.keys()),
        default="llama3.1-8b-base",
        help="Preset model name. Ignored if --repo-id is provided.",
    )
    parser.add_argument(
        "--repo-id",
        default="",
        help=f"HF repo id override. Default preset repo: {DEFAULT_REPO_ID}",
    )
    parser.add_argument(
        "--local-dir",
        default="",
        help="Directory to save model snapshot. Default is ../[model_name] from this script.",
    )
    return parser.parse_args()


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise ImportError(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    args = parse_args()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    repo_id = args.repo_id.strip() or PRESET_REPOS.get(args.model_preset, DEFAULT_REPO_ID)
    local_dir = resolve_local_dir(args.local_dir, repo_id)

    print(f"[INFO] Downloading model: {repo_id}")
    print(f"[INFO] Saving to: {local_dir}")

    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=local_dir,
        token=hf_token,
        local_dir_use_symlinks=False,
        resume_download=True,
    )


if __name__ == "__main__":
    main()
