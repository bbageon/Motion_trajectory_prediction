# python 06_finetune_llama2_lora.py \
#   --train-jsonl ./finetune_dataset_delta_train.jsonl \
#   --max-length 32768
#   --epochs 20 \
#   --learning-rate 2e-4 \
#   --batch-size 1 \
#   --grad-accum 4 \
#   --lora-r 32 \
#   --lora-alpha 64

import argparse
import inspect
import json
import os
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

STRUCT_TOKENS = ["[NUM]", "[INT]", "[DEC]", "[SEP]", "[ENDNUM]"]


def choose_output_dir(train_jsonl: str, user_output_dir: str) -> str:
    if user_output_dir:
        return user_output_dir
    name = os.path.basename(train_jsonl).lower()
    if "absolute" in name:
        return "./motionQA_finetuned_lora_mps"
    return "./motionQA_delta_finetuned_lora_mps"


def resolve_resume_checkpoint(resume_arg: str, output_dir: str) -> str:
    candidate = (resume_arg or "").strip()
    if not candidate:
        return ""
    if candidate.lower() not in {"last", "latest", "auto"}:
        return candidate

    out_dir = Path(output_dir)
    if not out_dir.exists():
        raise ValueError(f"Output dir does not exist: {out_dir}")

    checkpoints = []
    for p in out_dir.iterdir():
        if not p.is_dir() or not p.name.startswith("checkpoint-"):
            continue
        suffix = p.name.split("-", 1)[1]
        if suffix.isdigit():
            checkpoints.append((int(suffix), p))

    if not checkpoints:
        raise ValueError(f"No checkpoint-* directories found in: {out_dir}")

    checkpoints.sort(key=lambda x: x[0])
    return str(checkpoints[-1][1])


def format_example(prompt: str, completion: str) -> Tuple[str, str]:
    # 추론 코드(make_prompt_from_pose.py)와 동일한 포맷 사용
    prompt_text = prompt
    full_text = f"{prompt_text}\n{completion}"
    return prompt_text, full_text


class MotionJsonlDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int):
        self.rows: List[Dict[str, List[int]]] = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        skipped_empty = 0
        skipped_no_target = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                prompt = data.get("prompt", "").strip()
                completion = data.get("completion", "").strip()
                if not prompt or not completion:
                    skipped_empty += 1
                    continue

                prompt_text, _ = format_example(prompt, completion)
                prompt_ids = tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                )["input_ids"]

                # 핵심 정책:
                # 1) completion은 모델이 직접 맞춰야 하는 정답 구간이므로 먼저 확보한다.
                # 2) max_length 예산을 초과하면 prompt를 잘라서 completion 학습 신호를 보존한다.
                #    (prompt만 남고 completion이 사라지면 labels가 전부 -100이 되어 학습이 안 됨)
                completion_ids = tokenizer(
                    "\n" + completion,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=max_length - 1,
                )["input_ids"]
                if not completion_ids:
                    skipped_no_target += 1
                    continue

                # 전체 길이 제한:
                # [prompt_ids] + [completion_ids] + [EOS] <= max_length
                # 따라서 prompt에 할당 가능한 최대 길이를 계산한다.
                allowed_prompt_len = max_length - len(completion_ids) - 1
                if allowed_prompt_len < 0:
                    # completion 자체가 너무 길면 completion도 max_length-1까지만 유지한다.
                    # (EOS 1자리는 항상 남겨야 함)
                    completion_ids = completion_ids[: max_length - 1]
                    allowed_prompt_len = 0

                if allowed_prompt_len == 0:
                    prompt_ids = []
                elif len(prompt_ids) > allowed_prompt_len:
                    # 긴 prompt를 모두 넣을 수 없을 때는 뒤쪽 토큰을 유지한다.
                    # 현재 데이터 포맷은 관측 시퀀스가 뒤로 갈수록 최신 프레임에 가까워서
                    # 최근 맥락 보존이 유리하다.
                    prompt_ids = prompt_ids[-allowed_prompt_len:]

                eos_id = tokenizer.eos_token_id
                if eos_id is None:
                    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                if eos_id is None:
                    raise ValueError("Tokenizer EOS token/id is required for training.")

                full_ids = prompt_ids + completion_ids + [eos_id]
                labels = ([-100] * len(prompt_ids)) + completion_ids + [eos_id]

                self.rows.append(
                    {
                        "input_ids": full_ids,
                        "attention_mask": [1] * len(full_ids),
                        "labels": labels,
                    }
                )

        if not self.rows:
            raise ValueError(f"No valid training rows found in: {path}")
        print(
            f"[INFO] dataset loaded: kept={len(self.rows)}, "
            f"skipped_empty={skipped_empty}, skipped_no_target={skipped_no_target}"
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        item = self.rows[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(item["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
        }


@dataclass
class MotionDataCollator:
    pad_token_id: int

    def __call__(self, features):
        max_len = max(len(x["input_ids"]) for x in features)

        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            seq_len = len(f["input_ids"])
            pad_len = max_len - seq_len
            input_ids.append(
                torch.cat(
                    [
                        f["input_ids"],
                        torch.full((pad_len,), self.pad_token_id, dtype=torch.long),
                    ]
                )
            )
            attention_mask.append(
                torch.cat([f["attention_mask"], torch.zeros((pad_len,), dtype=torch.long)])
            )
            labels.append(
                torch.cat([f["labels"], torch.full((pad_len,), -100, dtype=torch.long)])
            )

        return {
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
            "labels": torch.stack(labels),
        }


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: str,
    learning_rate: float,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    save_steps: int,
    logging_steps: int,
    load_best_at_end: bool,
):
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    use_fp16 = False
    use_bf16 = device == "cuda"  # RTX 5090은 bf16 네이티브 지원

    # transformers 버전마다 TrainingArguments 파라미터명이 다를 수 있어
    # 시그니처를 확인해 지원되는 키만 전달한다.
    sig = inspect.signature(TrainingArguments.__init__).parameters
    has = lambda k: k in sig

    ta_kwargs = {
        "output_dir": output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": learning_rate,
        "num_train_epochs": epochs,
        "weight_decay": 0.01,
        "warmup_ratio": 0.03,
        "logging_steps": logging_steps,
        "save_steps": save_steps,
        "save_total_limit": 2,
        "fp16": use_fp16,
        "bf16": use_bf16,
        "gradient_checkpointing": True,
        "report_to": "none",
        "lr_scheduler_type": "cosine",
        "eval_steps": save_steps if eval_dataset is not None else None,
    }

    if has("overwrite_output_dir"):
        ta_kwargs["overwrite_output_dir"] = True
    if has("save_strategy"):
        ta_kwargs["save_strategy"] = "steps"
    if has("evaluation_strategy"):
        ta_kwargs["evaluation_strategy"] = "steps" if eval_dataset is not None else "no"
    elif has("eval_strategy"):
        ta_kwargs["eval_strategy"] = "steps" if eval_dataset is not None else "no"
    if eval_dataset is not None and load_best_at_end:
        if has("load_best_model_at_end"):
            ta_kwargs["load_best_model_at_end"] = True
        if has("metric_for_best_model"):
            ta_kwargs["metric_for_best_model"] = "eval_loss"
        if has("greater_is_better"):
            ta_kwargs["greater_is_better"] = False

    args = TrainingArguments(**ta_kwargs)

    model.config.use_cache = False
    collator = MotionDataCollator(pad_token_id=tokenizer.pad_token_id)

    return Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )


def resolve_trainable_special_token_ids(tokenizer) -> List[int]:
    ids: List[int] = []
    missing: List[str] = []
    for tok in STRUCT_TOKENS:
        tok_id = tokenizer.convert_tokens_to_ids(tok)
        if tok_id is None or tok_id < 0 or tok_id == tokenizer.unk_token_id:
            missing.append(tok)
            continue
        ids.append(tok_id)

    if missing:
        raise ValueError(
            "Special tokens are missing in tokenizer vocab: "
            f"{missing}. Run tokenizer extension first."
        )
    return sorted(set(ids))


def find_lm_head_weight_parameter(model) -> Tuple[str, torch.nn.Parameter]:
    out = model.get_output_embeddings()
    if out is not None and hasattr(out, "weight") and isinstance(out.weight, torch.nn.Parameter):
        return "output_embeddings.weight", out.weight

    candidates: List[Tuple[str, torch.nn.Parameter]] = []
    for name, param in model.named_parameters():
        if "lm_head" in name and name.endswith("weight"):
            candidates.append((name, param))
    if not candidates:
        raise RuntimeError("Could not find lm_head weight parameter.")

    # Prefer a trainable candidate, then larger tensor.
    candidates.sort(key=lambda x: (not x[1].requires_grad, -x[1].numel()))
    return candidates[0]


def attach_row_grad_mask(param: torch.nn.Parameter, trainable_rows: List[int]) -> None:
    if param.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix for row masking, got shape={tuple(param.shape)}")
    if not trainable_rows:
        raise ValueError("trainable_rows must not be empty.")
    if max(trainable_rows) >= param.shape[0]:
        raise ValueError(
            f"Token id out of range for lm_head rows: max_id={max(trainable_rows)}, rows={param.shape[0]}"
        )

    row_mask = torch.zeros((param.shape[0], 1), dtype=torch.float32)
    row_mask[trainable_rows] = 1.0
    cache: Dict[Tuple[str, str], torch.Tensor] = {}

    def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
        key = (str(grad.device), str(grad.dtype))
        mask = cache.get(key)
        if mask is None:
            mask = row_mask.to(device=grad.device, dtype=grad.dtype)
            cache[key] = mask
        return grad * mask

    param.register_hook(_mask_grad)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA finetuning for local Llama2 on motion JSONL.")
    parser.add_argument("--base-model-dir", default="../Meta-Llama-3.1-8B_tokenizerExtension")
    parser.add_argument("--train-jsonl", default="./finetune_dataset_delta.jsonl")
    parser.add_argument(
        "--val-jsonl",
        default="",
        help="Optional validation JSONL. If set, this is used for eval instead of random split.",
    )
    parser.add_argument("--output-dir", default="")
    # --max-length는 "한 샘플(프롬프트+정답+EOS)"의 최대 토큰 길이.
    # 값을 키우면 잘림은 줄지만, 메모리 사용량이 크게 증가한다.
    # Transformer attention 비용은 길이에 대해 대략 O(L^2)로 증가하므로
    # 길이를 2배로 키우면 메모리/연산량이 4배 수준으로 뛸 수 있다.
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--epochs", type=float, default=6.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--resume-from-checkpoint",
        default="",
        help="Checkpoint path to resume from. Use 'last' to auto-pick latest checkpoint-* in output-dir.",
    )
    parser.add_argument(
        "--no-load-best-at-end",
        action="store_true",
        help="Disable automatic loading of the best checkpoint based on eval_loss.",
    )
    parser.add_argument(
        "--no-train-token-io",
        action="store_true",
        help="Disable training/saving embed_tokens and lm_head with LoRA.",
    )
    parser.add_argument(
        "--token-io-mode",
        choices=["full", "special_only", "lm_head_only", "lm_head_special_only", "off"],
        default="full",
        help=(
            "How to train token I/O layers: "
            "'full'=train full embed/lm_head, "
            "'special_only'=train only added special token rows, "
            "'lm_head_only'=train full lm_head only, "
            "'lm_head_special_only'=train only special token rows in lm_head, "
            "'off'=freeze token I/O."
        ),
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    # 상대경로 인자는 현재 작업 디렉토리가 아니라 스크립트 위치 기준으로 해석한다.
    args.base_model_dir = str((script_dir / args.base_model_dir).resolve()) if not os.path.isabs(args.base_model_dir) else args.base_model_dir
    args.train_jsonl = str((script_dir / args.train_jsonl).resolve()) if not os.path.isabs(args.train_jsonl) else args.train_jsonl
    if args.val_jsonl and not os.path.isabs(args.val_jsonl):
        args.val_jsonl = str((script_dir / args.val_jsonl).resolve())
    if args.output_dir and not os.path.isabs(args.output_dir):
        args.output_dir = str((script_dir / args.output_dir).resolve())

    if not os.path.exists(args.base_model_dir):
        raise FileNotFoundError(f"base model dir not found: {args.base_model_dir}")
    if not os.path.exists(args.train_jsonl):
        raise FileNotFoundError(f"train jsonl not found: {args.train_jsonl}")
    if args.val_jsonl and not os.path.exists(args.val_jsonl):
        raise FileNotFoundError(f"val jsonl not found: {args.val_jsonl}")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.no_train_token_io:
        args.token_io_mode = "off"

    output_dir = choose_output_dir(args.train_jsonl, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Loading tokenizer/model from: {args.base_model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        local_files_only=True,
        torch_dtype=torch.bfloat16,  # fp32(32GB) → bf16(16GB), RTX 5090 bf16 지원
    )

    modules_to_save = None
    trainable_token_indices = None
    ensure_weight_tying = False
    special_token_ids = resolve_trainable_special_token_ids(tokenizer)
    if args.token_io_mode == "full":
        modules_to_save = ["embed_tokens", "lm_head"]
        ensure_weight_tying = True
    elif args.token_io_mode == "special_only":
        trainable_token_indices = special_token_ids
        ensure_weight_tying = bool(getattr(model.config, "tie_word_embeddings", False))
        if not ensure_weight_tying:
            print(
                "[WARN] tie_word_embeddings=False; special_only mode updates input embedding rows only."
            )
    elif args.token_io_mode == "lm_head_only":
        modules_to_save = ["lm_head"]
    elif args.token_io_mode == "lm_head_special_only":
        modules_to_save = ["lm_head"]

    lora_sig = inspect.signature(LoraConfig.__init__).parameters
    lora_kwargs = {
        "task_type": TaskType.CAUSAL_LM,
        "r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "modules_to_save": modules_to_save,
    }
    if trainable_token_indices is not None:
        if "trainable_token_indices" not in lora_sig:
            raise RuntimeError(
                "Current peft version does not support trainable_token_indices. "
                "Upgrade peft or use --token-io-mode full/off."
            )
        lora_kwargs["trainable_token_indices"] = trainable_token_indices
    if ensure_weight_tying and "ensure_weight_tying" in lora_sig:
        lora_kwargs["ensure_weight_tying"] = True

    lora_cfg = LoraConfig(**lora_kwargs)
    model = get_peft_model(model, lora_cfg)
    if args.token_io_mode == "lm_head_special_only":
        lm_head_name, lm_head_weight = find_lm_head_weight_parameter(model)
        attach_row_grad_mask(lm_head_weight, special_token_ids)
        print(
            "[INFO] lm_head special-row mask enabled: "
            f"module={lm_head_name}, token_ids={special_token_ids}"
        )
    if args.token_io_mode == "off":
        print("[INFO] token io train disabled: embed_tokens/lm_head are frozen.")
    elif args.token_io_mode == "special_only":
        print(
            "[INFO] token io train mode=special_only: "
            f"trainable token ids={trainable_token_indices}"
        )
    elif args.token_io_mode == "lm_head_only":
        print("[INFO] token io train mode=lm_head_only: lm_head full matrix will be updated.")
    elif args.token_io_mode == "lm_head_special_only":
        print(
            "[INFO] token io train mode=lm_head_special_only: "
            "only special-token rows in lm_head get non-zero gradients."
        )
    else:
        print("[INFO] token io train enabled: embed_tokens/lm_head will be updated and saved.")
    model.print_trainable_parameters()

    dataset = MotionJsonlDataset(args.train_jsonl, tokenizer, args.max_length)
    total = len(dataset)
    print(f"[INFO] total samples: {total}")

    eval_dataset = None
    train_dataset = dataset

    # 1) val-jsonl이 있으면 random split 대신 명시적 검증셋을 사용한다.
    if args.val_jsonl:
        eval_dataset = MotionJsonlDataset(args.val_jsonl, tokenizer, args.max_length)
        print(
            f"[INFO] train samples: {len(train_dataset)}, "
            f"val samples(from --val-jsonl): {len(eval_dataset)}"
        )
    # 2) 없으면 train 내부 random split을 사용한다.
    elif 0.0 < args.val_ratio < 1.0 and total >= 10:
        val_size = max(1, int(total * args.val_ratio))
        train_size = total - val_size
        train_dataset, eval_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        print(f"[INFO] train samples: {len(train_dataset)}, val samples: {len(eval_dataset)}")
    else:
        print("[INFO] validation split skipped")

    if eval_dataset is None and not args.no_load_best_at_end:
        raise ValueError(
            "Best-checkpoint selection needs validation data. "
            "Set --val-jsonl <path> or use --val-ratio > 0 with enough samples, "
            "or pass --no-load-best-at-end."
        )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        load_best_at_end=not args.no_load_best_at_end,
    )

    resume_checkpoint = resolve_resume_checkpoint(args.resume_from_checkpoint, output_dir)
    print("[INFO] Start training")
    if resume_checkpoint:
        print(f"[INFO] Resuming from checkpoint: {resume_checkpoint}")
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    if eval_dataset is not None and not args.no_load_best_at_end:
        if trainer.state.best_model_checkpoint is None:
            raise RuntimeError(
                "Best checkpoint was not selected. Check TrainingArguments compatibility and eval setup."
            )
        print(
            f"[INFO] Best checkpoint: {trainer.state.best_model_checkpoint}, "
            f"best eval_loss: {trainer.state.best_metric}"
        )

    print(f"[INFO] Saving LoRA adapter to: {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("[INFO] Done")


if __name__ == "__main__":
    main()
