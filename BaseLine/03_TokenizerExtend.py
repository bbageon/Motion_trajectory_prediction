# python BaseLine\03_TokenizerExtend.py --model-preset llama3.1-8b-base

from __future__ import annotations

import argparse
from pathlib import Path

MODEL_PRESETS = {
    "llama2-7b": "Llama-2-7b-chat-hf",
    "llama2-13b": "Llama-2-13b-chat-hf",
    "llama3.1-8b-base": "Meta-Llama-3.1-8B",
    "llama3.1-8b-instruct": "Meta-Llama-3.1-8B-Instruct",
}

STRUCT_TOKENS = ["[NUM]", "[INT]", "[DEC]", "[SEP]", "[ENDNUM]"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extend tokenizer/model with motion structure tokens.")
    parser.add_argument(
        "--model-preset",
        choices=list(MODEL_PRESETS.keys()),
        default="llama3.1-8b-base",
        help="Choose local base model preset.",
    )
    parser.add_argument(
        "--model-path",
        default="",
        help="Local model path override. Default is ../[preset_model_name] from this script.",
    )
    parser.add_argument(
        "--save-dir",
        default="",
        help="Output path override. Default is ../[preset_model_name]_tokenizerExtension from this script.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[str, str]:
    script_dir = Path(__file__).resolve().parent
    preset_name = MODEL_PRESETS[args.model_preset]

    model_path = args.model_path.strip()
    if not model_path:
        model_path = str((script_dir / ".." / preset_name).resolve())

    save_dir = args.save_dir.strip()
    if not save_dir:
        save_dir = str((script_dir / ".." / f"{preset_name}_tokenizerExtension").resolve())

    return model_path, save_dir


def main() -> None:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise ImportError(
            "transformers and torch are required. Install with: pip install transformers torch"
        ) from e

    args = parse_args()
    model_path, save_dir = resolve_paths(args)

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    print(f"Loading local tokenizer/model from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_kwargs = {"local_files_only": True, "low_cpu_mem_usage": True}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float16
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    old_vocab_size = len(tokenizer)

    existing = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    merged = list(dict.fromkeys(existing + STRUCT_TOKENS))
    added = tokenizer.add_special_tokens({"additional_special_tokens": merged})

    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    print(f"Old vocab size: {old_vocab_size}")
    print(f"Added tokens: {added}")
    print(f"New vocab size: {len(tokenizer)}")
    for token in STRUCT_TOKENS:
        print(f"{token} -> {tokenizer.convert_tokens_to_ids(token)}")

    tokenizer.save_pretrained(save_dir)
    model.save_pretrained(
        save_dir,
        max_shard_size="2GB",
        save_original_format=True,
    )
    print(f"Saved to: {save_dir}")


if __name__ == "__main__":
    main()
