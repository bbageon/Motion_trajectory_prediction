import os
from huggingface_hub import snapshot_download

REPO_ID = "meta-llama/Llama-2-7b-chat-hf"
LOCAL_DIR = "./llama2_local"


def main() -> None:
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN environment variable is not set.")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        local_dir=LOCAL_DIR,
        token=hf_token,
        local_dir_use_symlinks=False,
    )


if __name__ == "__main__":
    main()
