from __future__ import annotations

from transformers import AutoModelForCausalLM, AutoTokenizer

# 1) 원본 로컬 모델/토크나이저 경로
MODEL_PATH = "./llama2_local"

# 2) 확장 결과를 저장할 경로 (원본 보존)
SAVE_DIR = "./llama2_local_tokenizerExtension"

# 3) 숫자 구조를 명시하기 위한 사용자 정의 토큰
STRUCT_TOKENS = ["[NUM]", "[INT]", "[DEC]", "[SEP]", "[ENDNUM]"]


def main() -> None:
    # A. 원본 모델/토크나이저를 로컬에서 로드한다.
    print("Loading local tokenizer/model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)

    # B. 확장 전 vocab 크기를 기록해 비교에 사용한다.
    old_vocab_size = len(tokenizer)

    # C. 기존 additional_special_tokens + 구조 토큰을 병합(중복 제거)한다.
    #    dict.fromkeys(...)를 사용해 순서를 유지한 채 중복만 제거한다.
    existing = tokenizer.special_tokens_map.get("additional_special_tokens", [])
    merged = list(dict.fromkeys(existing + STRUCT_TOKENS))
    added = tokenizer.add_special_tokens({"additional_special_tokens": merged})

    # D. 토큰이 실제로 추가되면 모델 임베딩 크기도 vocab에 맞게 확장한다.
    #    (미동기화 상태면 forward에서 index 오류가 날 수 있다.)
    if added > 0:
        model.resize_token_embeddings(len(tokenizer))

    # E. 확장 결과를 로그로 출력한다.
    print(f"Old vocab size: {old_vocab_size}")
    print(f"Added tokens: {added}")
    print(f"New vocab size: {len(tokenizer)}")
    for token in STRUCT_TOKENS:
        print(f"{token} -> {tokenizer.convert_tokens_to_ids(token)}")

    # F. 확장된 토크나이저/모델을 새 경로에 저장한다.
    #    max_shard_size로 대형 모델 저장 시 파일을 여러 개로 분할한다.
    tokenizer.save_pretrained(SAVE_DIR)
    model.save_pretrained(
        SAVE_DIR,
        max_shard_size="2GB",
        save_original_format=True,
    )
    print(f"Saved to: {SAVE_DIR}")


if __name__ == "__main__":
    main()
